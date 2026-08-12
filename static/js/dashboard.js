/**
 * Dashboard SSE Consumer and Attack Log Terminal Panel.
 *
 * Handles:
 *  - SSE connection to /events with exponential backoff reconnect
 *  - Attack log terminal panel (formatting, color-coding, FIFO cap, typing cursor)
 *  - Dispatches events to GlobeManager for arc visualization
 *
 * Requirements: 8.1, 8.2, 8.3, 8.4, 4.1, 4.2, 15.2, 15.3
 */
(function () {
    'use strict';

    // =========================================================================
    // CONFIGURATION
    // =========================================================================

    var LOG_MAX_ENTRIES = 200;
    var SSE_RECONNECT_BASE_MS = 1000;
    var SSE_RECONNECT_MAX_MS = 30000;

    // Attack type color mapping (mirrors globe.js ATTACK_TYPE_COLORS)
    var ATTACK_TYPE_COLORS = {
        'SYN Flood': '#FF0040',
        'UDP Flood': '#FF8C00',
        'DNS Amplification': '#9B59B6',
        'HTTP Flood': '#FFD700',
        'LDAP': '#00E5FF',
        'NTP': '#00E5FF',
        'MSSQL': '#00E5FF',
        'NetBIOS': '#00FF41',
        'SSDP': '#00FF41',
        'TFTP': '#00FF41',
        'UDPLag': '#00FF41',
        'WebDDoS': '#FFD700'
    };

    var DEFAULT_COLOR = '#00FF41';

    // =========================================================================
    // STATE
    // =========================================================================

    var logContainer = null;
    var logCountEl = null;
    var logEntryCount = 0;
    var logQueue = [];
    var rafScheduled = false;
    var eventSource = null;
    var reconnectDelay = SSE_RECONNECT_BASE_MS;
    var cursorEl = null;

    // =========================================================================
    // IP MASKING
    // =========================================================================

    /**
     * Mask the last two octets of an IPv4 address.
     * "185.220.101.34" → "185.220.x.x"
     *
     * @param {string} ip - IPv4 address string
     * @returns {string} Masked IP
     */
    function maskIp(ip) {
        if (!ip || typeof ip !== 'string') return 'x.x.x.x';
        var parts = ip.split('.');
        if (parts.length !== 4) return ip;
        return parts[0] + '.' + parts[1] + '.x.x';
    }

    // =========================================================================
    // LOG ENTRY CREATION
    // =========================================================================

    /**
     * Create a formatted log entry DOM element from an attack event.
     *
     * Format: [HH:MM:SS] ATTACK_TYPE  IP_MASKED  COUNTRY  CONF%  TIME_MS
     *
     * @param {Object} event - Enhanced attack event
     * @returns {HTMLElement} Log entry div
     */
    function createLogEntry(event) {
        var el = document.createElement('div');
        el.className = 'log-entry';
        el.setAttribute('data-type', event.attack_type || '');

        // Mark live events distinctly
        if (event.source_channel === 'live') {
            el.classList.add('live-event');
        }

        // Format timestamp (HH:MM:SS)
        var ts = '';
        if (event.timestamp) {
            try {
                var d = new Date(event.timestamp);
                ts = d.toTimeString().slice(0, 8);
            } catch (e) {
                ts = '--:--:--';
            }
        } else {
            var now = new Date();
            ts = now.toTimeString().slice(0, 8);
        }

        // Confidence percentage
        var confPct = event.confidence != null
            ? (event.confidence * 100).toFixed(1) + '%'
            : '—';

        // Classification time
        var classTime = event.classified_in_ms != null
            ? event.classified_in_ms.toFixed(1) + 'ms'
            : '—';

        // Masked IP
        var maskedIp = maskIp(event.ip_address);

        // Country
        var country = event.country || '??';

        // Attack type color
        var typeColor = ATTACK_TYPE_COLORS[event.attack_type] || DEFAULT_COLOR;

        // Build inner HTML with spans for styling
        el.innerHTML =
            '<span class="log-time">' + ts + '</span>' +
            '<span class="log-type" style="color:' + typeColor + '">' + (event.attack_type || 'UNKNOWN') + '</span>' +
            '<span class="log-ip">' + maskedIp + '</span>' +
            '<span class="log-country">' + country + '</span>' +
            '<span class="log-confidence">' + confPct + '</span>' +
            '<span class="log-classtime">' + classTime + '</span>';

        return el;
    }

    // =========================================================================
    // LOG QUEUE & RAF-BATCHED DOM UPDATES
    // =========================================================================

    /**
     * Queue a log entry for batched DOM insertion.
     * Uses requestAnimationFrame to avoid layout thrashing.
     *
     * @param {Object} event - Enhanced attack event
     */
    function queueLogEntry(event) {
        logQueue.push(event);
        if (!rafScheduled) {
            rafScheduled = true;
            requestAnimationFrame(flushLogQueue);
        }
    }

    /**
     * Flush queued log entries into the DOM.
     * Prepends new entries (newest at top) and enforces 200-entry FIFO cap.
     */
    function flushLogQueue() {
        rafScheduled = false;
        if (!logContainer || logQueue.length === 0) return;

        // Create a document fragment for batch DOM insertion
        var frag = document.createDocumentFragment();
        var entries = logQueue.splice(0, logQueue.length);

        for (var i = 0; i < entries.length; i++) {
            frag.appendChild(createLogEntry(entries[i]));
        }

        // Prepend new entries at the top
        if (logContainer.firstChild) {
            logContainer.insertBefore(frag, logContainer.firstChild);
        } else {
            logContainer.appendChild(frag);
        }

        // Update entry count
        logEntryCount += entries.length;

        // FIFO cap: remove oldest entries (from the bottom) exceeding 200
        while (logEntryCount > LOG_MAX_ENTRIES) {
            var last = logContainer.lastChild;
            // Don't remove the cursor element
            if (last && last !== cursorEl) {
                logContainer.removeChild(last);
                logEntryCount--;
            } else {
                break;
            }
        }

        // Update entry count display
        if (logCountEl) {
            logCountEl.textContent = logEntryCount + ' entries';
        }
    }

    // =========================================================================
    // TYPING CURSOR (BLINKING TERMINAL CURSOR)
    // =========================================================================

    /**
     * Create and append the blinking terminal cursor element.
     * The typing effect is achieved naturally by the stream pace (10-50 events/sec).
     * The cursor provides the visual "actively typing" indicator.
     */
    function initCursor() {
        if (!logContainer) return;
        cursorEl = document.createElement('span');
        cursorEl.className = 'log-cursor';
        cursorEl.setAttribute('aria-hidden', 'true');
        // Place cursor at the top of the log container
        if (logContainer.firstChild) {
            logContainer.insertBefore(cursorEl, logContainer.firstChild);
        } else {
            logContainer.appendChild(cursorEl);
        }
    }

    /**
     * Keep cursor at the top (before newest entries) after each flush.
     */
    function repositionCursor() {
        if (!cursorEl || !logContainer) return;
        if (logContainer.firstChild !== cursorEl) {
            logContainer.insertBefore(cursorEl, logContainer.firstChild);
        }
    }

    // Patch flushLogQueue to reposition cursor after DOM insert
    var originalFlush = flushLogQueue;
    flushLogQueue = function () {
        originalFlush();
        repositionCursor();
    };

    // =========================================================================
    // SSE CONNECTION WITH EXPONENTIAL BACKOFF
    // =========================================================================

    /**
     * Connect to the SSE event stream.
     * On receiving events, dispatches to:
     *   - GlobeManager.addArc (globe visualization)
     *   - Attack log panel (this module)
     */
    function connectSSE() {
        if (eventSource) {
            eventSource.close();
        }

        eventSource = new EventSource('/events');

        eventSource.onopen = function () {
            // Reset reconnect delay on successful connection
            reconnectDelay = SSE_RECONNECT_BASE_MS;
        };

        // Listen for named "attack" events (server sends event: attack)
        eventSource.addEventListener('attack', function (e) {
            try {
                var event = JSON.parse(e.data);
                handleEvent(event);
            } catch (err) {
                console.error('[dashboard] Failed to parse SSE event:', err);
            }
        });

        // Also listen for unnamed messages (backward compat)
        eventSource.onmessage = function (e) {
            try {
                var event = JSON.parse(e.data);
                handleEvent(event);
            } catch (err) {
                // Ignore parse errors on generic messages (heartbeats etc)
            }
        };

        eventSource.onerror = function () {
            eventSource.close();
            eventSource = null;
            // Exponential backoff reconnect
            setTimeout(function () {
                reconnectDelay = Math.min(reconnectDelay * 2, SSE_RECONNECT_MAX_MS);
                connectSSE();
            }, reconnectDelay);
        };
    }

    /**
     * Handle a single SSE event — dispatch to globe, log panel, and timeline.
     *
     * @param {Object} event - Parsed enhanced attack event
     */
    function handleEvent(event) {
        // Dispatch to globe for arc visualization
        if (window.GlobeManager && window.GlobeManager.addArc) {
            window.GlobeManager.addArc(event);
        }

        // Feed timeline with real-time event data
        if (window.PanelsManager && window.PanelsManager.recordTimelineEvent && event.attack_type) {
            window.PanelsManager.recordTimelineEvent(event.attack_type);
        }

        // Queue for attack log panel
        queueLogEntry(event);
    }

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    function init() {
        logContainer = document.getElementById('log-entries');
        logCountEl = document.getElementById('log-count');

        if (!logContainer) {
            console.error('[dashboard] #log-entries container not found');
            return;
        }

        // Initialize the blinking cursor
        initCursor();

        // Connect to SSE stream
        connectSSE();
    }

    // =========================================================================
    // PUBLIC API
    // =========================================================================

    window.DashboardManager = {
        /** Manually add a log entry (for testing) */
        addLogEntry: queueLogEntry,

        /** Get current log entry count */
        getLogEntryCount: function () { return logEntryCount; },

        /** Exposed utilities for testing */
        maskIp: maskIp,
        createLogEntry: createLogEntry,

        /** Constants */
        LOG_MAX_ENTRIES: LOG_MAX_ENTRIES,
        ATTACK_TYPE_COLORS: ATTACK_TYPE_COLORS
    };

    // =========================================================================
    // BOOTSTRAP
    // =========================================================================

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
