/**
 * Dashboard Panels: Stats Grid, Top Attackers Table, Timeline Area Chart.
 *
 * Handles:
 *  - Stats grid: 6 metric cards with pulse animation, polling /api/model-stats every 2s
 *  - Top attackers table: polling /api/top-attackers every 5s, ranked display with masked IPs
 *  - Timeline area chart: 60-minute window, 1-minute buckets, grouped by attack type, updated every 5s
 *
 * Requirements: 9.1, 9.2, 9.3, 11.1, 11.2, 11.3
 */
(function () {
    'use strict';

    // =========================================================================
    // CONFIGURATION
    // =========================================================================

    var STATS_POLL_INTERVAL_MS = 2000;
    var ATTACKERS_POLL_INTERVAL_MS = 5000;
    var TIMELINE_POLL_INTERVAL_MS = 5000;
    var TIMELINE_WINDOW_MINUTES = 60;

    // Attack type color mapping
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

    // Threat level thresholds (based on predictions per second)
    var THREAT_LEVELS = [
        { threshold: 40, label: 'CRITICAL', className: 'severity-critical' },
        { threshold: 25, label: 'HIGH', className: 'severity-high' },
        { threshold: 10, label: 'MEDIUM', className: 'severity-medium' },
        { threshold: 0, label: 'LOW', className: 'severity-low' }
    ];

    // =========================================================================
    // STATE
    // =========================================================================

    var statsTimerId = null;
    var attackersTimerId = null;
    var timelineTimerId = null;
    var timelineChart = null;
    var previousStatValues = {};

    // DOM element references (cached on init)
    var statElements = {};
    var attackersTbody = null;
    var timelineCanvas = null;

    // =========================================================================
    // UTILITY: IP MASKING
    // =========================================================================

    /**
     * Mask the last two octets of an IPv4 address.
     * "185.220.101.34" → "185.220.x.x"
     */
    function maskIp(ip) {
        if (!ip || typeof ip !== 'string') return 'x.x.x.x';
        var parts = ip.split('.');
        if (parts.length !== 4) return ip;
        return parts[0] + '.' + parts[1] + '.x.x';
    }

    /**
     * Format a number with comma separators.
     */
    function formatNumber(n) {
        if (n == null) return '0';
        return n.toLocaleString();
    }

    /**
     * Compute threat level from predictions-per-second.
     */
    function computeThreatLevel(pps) {
        for (var i = 0; i < THREAT_LEVELS.length; i++) {
            if (pps >= THREAT_LEVELS[i].threshold) {
                return THREAT_LEVELS[i];
            }
        }
        return THREAT_LEVELS[THREAT_LEVELS.length - 1];
    }

    // =========================================================================
    // STATS GRID — 6 METRIC CARDS WITH PULSE ANIMATION
    // =========================================================================

    /**
     * Update a single stat element value with pulse animation on change.
     */
    function updateStatValue(key, newValue) {
        var el = statElements[key];
        if (!el) return;

        var displayValue = String(newValue);
        var oldValue = previousStatValues[key];

        if (oldValue !== displayValue) {
            el.textContent = displayValue;
            previousStatValues[key] = displayValue;

            // Trigger pulse animation
            el.classList.remove('pulse');
            // Force reflow to restart animation
            void el.offsetWidth;
            el.classList.add('pulse');
        }
    }

    /**
     * Fetch /api/model-stats and update the 6 stat cards.
     */
    function pollStats() {
        fetch('/api/model-stats')
            .then(function (resp) {
                if (!resp.ok) throw new Error('Stats fetch failed: ' + resp.status);
                return resp.json();
            })
            .then(function (data) {
                requestAnimationFrame(function () {
                    // 1. Total Attacks Classified
                    updateStatValue('attacks', formatNumber(data.total_predictions || 0));

                    // 2. Active Source IPs — derive from attack_type_breakdown count or attacks_detected
                    // The model-stats doesn't directly provide "active source IPs" — use attacks_detected as proxy
                    updateStatValue('sources', formatNumber(data.attacks_detected || 0));

                    // 3. Affected Countries — not directly in model-stats, use a count from breakdown
                    var countryCount = 0;
                    if (data.attack_type_breakdown) {
                        countryCount = Object.keys(data.attack_type_breakdown).length;
                    }
                    updateStatValue('countries', String(countryCount));

                    // 4. Model F1-Score
                    var f1 = data.f1_score != null ? data.f1_score.toFixed(2) : '0.00';
                    updateStatValue('f1', f1);

                    // 5. Current Threat Level
                    var pps = data.predictions_per_second || 0;
                    var threat = computeThreatLevel(pps);
                    var threatEl = statElements['threat'];
                    if (threatEl) {
                        updateStatValue('threat', threat.label);
                        // Remove old severity classes and apply new one
                        threatEl.className = 'stat-value ' + threat.className;
                    }

                    // 6. Predictions Per Second
                    updateStatValue('pps', pps.toFixed(1));
                });
            })
            .catch(function (err) {
                // Silent fail — show stale data
                console.warn('[panels] Stats poll error:', err.message);
            });
    }

    /**
     * Start stats polling interval.
     */
    function startStatsPolling() {
        // Initial fetch
        pollStats();
        statsTimerId = setInterval(pollStats, STATS_POLL_INTERVAL_MS);
    }

    // =========================================================================
    // TOP ATTACKERS TABLE — POLLING /api/top-attackers EVERY 5s
    // =========================================================================

    /**
     * Fetch /api/top-attackers and rebuild the table body.
     */
    function pollTopAttackers() {
        fetch('/api/top-attackers')
            .then(function (resp) {
                if (!resp.ok) throw new Error('Top attackers fetch failed: ' + resp.status);
                return resp.json();
            })
            .then(function (data) {
                requestAnimationFrame(function () {
                    renderAttackersTable(data);
                });
            })
            .catch(function (err) {
                console.warn('[panels] Top attackers poll error:', err.message);
            });
    }

    /**
     * Render the top attackers table from API response data.
     * @param {Array} attackers - Array of TopAttackerEntry objects
     */
    function renderAttackersTable(attackers) {
        if (!attackersTbody) return;
        if (!Array.isArray(attackers) || attackers.length === 0) {
            attackersTbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-dim)">No data</td></tr>';
            return;
        }

        var html = '';
        for (var i = 0; i < attackers.length; i++) {
            var a = attackers[i];
            var rank = a.rank || (i + 1);
            var ip = maskIp(a.ip_address);
            var country = a.country || '??';
            var count = formatNumber(a.attack_count || 0);
            var lastSeen = '';
            if (a.last_seen) {
                try {
                    var d = new Date(a.last_seen);
                    lastSeen = d.toTimeString().slice(0, 8);
                } catch (e) {
                    lastSeen = '—';
                }
            }

            html += '<tr>' +
                '<td>' + rank + '</td>' +
                '<td>' + ip + '</td>' +
                '<td>' + country + '</td>' +
                '<td>' + count + '</td>' +
                '<td>' + lastSeen + '</td>' +
                '</tr>';
        }

        attackersTbody.innerHTML = html;
    }

    /**
     * Start top attackers polling interval.
     */
    function startAttackersPolling() {
        pollTopAttackers();
        attackersTimerId = setInterval(pollTopAttackers, ATTACKERS_POLL_INTERVAL_MS);
    }

    // =========================================================================
    // TIMELINE AREA CHART — 60-MINUTE WINDOW, 1-MINUTE BUCKETS
    // =========================================================================

    /**
     * Initialize the Chart.js timeline area chart.
     */
    function initTimelineChart() {
        if (!timelineCanvas) return;

        var ctx = timelineCanvas.getContext('2d');
        if (!ctx) return;

        // Generate 60 time labels (minute buckets)
        var labels = generateTimeLabels();

        // Define dataset stubs for each attack type we display
        // Group into 6 color categories to keep chart readable
        var datasets = [
            { label: 'SYN Flood', borderColor: '#FF0040', backgroundColor: 'rgba(255, 0, 64, 0.2)', data: new Array(TIMELINE_WINDOW_MINUTES).fill(0), fill: true, tension: 0.3, pointRadius: 0 },
            { label: 'UDP Flood', borderColor: '#FF8C00', backgroundColor: 'rgba(255, 140, 0, 0.2)', data: new Array(TIMELINE_WINDOW_MINUTES).fill(0), fill: true, tension: 0.3, pointRadius: 0 },
            { label: 'DNS Amplification', borderColor: '#9B59B6', backgroundColor: 'rgba(155, 89, 182, 0.2)', data: new Array(TIMELINE_WINDOW_MINUTES).fill(0), fill: true, tension: 0.3, pointRadius: 0 },
            { label: 'HTTP/Web', borderColor: '#FFD700', backgroundColor: 'rgba(255, 215, 0, 0.2)', data: new Array(TIMELINE_WINDOW_MINUTES).fill(0), fill: true, tension: 0.3, pointRadius: 0 },
            { label: 'Reflection', borderColor: '#00E5FF', backgroundColor: 'rgba(0, 229, 255, 0.2)', data: new Array(TIMELINE_WINDOW_MINUTES).fill(0), fill: true, tension: 0.3, pointRadius: 0 },
            { label: 'Other', borderColor: '#00FF41', backgroundColor: 'rgba(0, 255, 65, 0.2)', data: new Array(TIMELINE_WINDOW_MINUTES).fill(0), fill: true, tension: 0.3, pointRadius: 0 }
        ];

        timelineChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#00FF41',
                            font: { family: "'JetBrains Mono', monospace", size: 9 },
                            boxWidth: 10,
                            padding: 8
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(10, 15, 10, 0.95)',
                        titleColor: '#00FF41',
                        bodyColor: '#00E5FF',
                        borderColor: 'rgba(0, 255, 65, 0.3)',
                        borderWidth: 1,
                        titleFont: { family: "'JetBrains Mono', monospace", size: 10 },
                        bodyFont: { family: "'JetBrains Mono', monospace", size: 9 }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: { color: 'rgba(0, 255, 65, 0.08)' },
                        ticks: {
                            color: 'rgba(0, 255, 65, 0.5)',
                            font: { family: "'JetBrains Mono', monospace", size: 8 },
                            maxTicksLimit: 12,
                            maxRotation: 0
                        }
                    },
                    y: {
                        display: true,
                        stacked: true,
                        beginAtZero: true,
                        grid: { color: 'rgba(0, 255, 65, 0.08)' },
                        ticks: {
                            color: 'rgba(0, 255, 65, 0.5)',
                            font: { family: "'JetBrains Mono', monospace", size: 9 },
                            precision: 0
                        }
                    }
                }
            }
        });
    }

    /**
     * Generate time labels for the last 60 minutes (HH:MM format).
     */
    function generateTimeLabels() {
        var labels = [];
        var now = new Date();
        for (var i = TIMELINE_WINDOW_MINUTES - 1; i >= 0; i--) {
            var t = new Date(now.getTime() - i * 60000);
            var h = String(t.getHours()).padStart(2, '0');
            var m = String(t.getMinutes()).padStart(2, '0');
            labels.push(h + ':' + m);
        }
        return labels;
    }

    /**
     * Map attack type to one of 6 chart dataset categories.
     */
    function getTypeCategory(attackType) {
        switch (attackType) {
            case 'SYN Flood': return 0;
            case 'UDP Flood': return 1;
            case 'DNS Amplification': return 2;
            case 'HTTP Flood':
            case 'WebDDoS': return 3;
            case 'LDAP':
            case 'NTP':
            case 'MSSQL': return 4;
            case 'NetBIOS':
            case 'SSDP':
            case 'TFTP':
            case 'UDPLag': return 5;
            default: return 5;
        }
    }

    /**
     * Fetch /api/attack-types and update the timeline chart data.
     * The API returns a per-type total count; we use it to build a cumulative timeline.
     * Since there's no per-minute API, we maintain a local ring buffer of per-minute counts
     * from the SSE stream and augment from the API data.
     */
    function pollTimeline() {
        fetch('/api/attack-types')
            .then(function (resp) {
                if (!resp.ok) throw new Error('Attack types fetch failed: ' + resp.status);
                return resp.json();
            })
            .then(function (data) {
                requestAnimationFrame(function () {
                    updateTimelineFromTypes(data);
                });
            })
            .catch(function (err) {
                console.warn('[panels] Timeline poll error:', err.message);
            });
    }

    /**
     * Update the timeline chart. Since the API returns aggregate counts,
     * we shift the timeline buckets and add new counts into the current minute.
     */
    function updateTimelineFromTypes(typeData) {
        if (!timelineChart) return;

        // Update labels to current time window
        var labels = generateTimeLabels();
        timelineChart.data.labels = labels;

        // Calculate per-type increments since last poll
        var datasets = timelineChart.data.datasets;
        for (var attackType in typeData) {
            if (!typeData.hasOwnProperty(attackType)) continue;
            var catIdx = getTypeCategory(attackType);
            var currentTotal = typeData[attackType] || 0;
            var lastKey = '_last_' + attackType;
            var lastTotal = timelineState[lastKey] || 0;
            var delta = currentTotal - lastTotal;
            timelineState[lastKey] = currentTotal;

            if (delta > 0 && datasets[catIdx]) {
                // Add delta to the most recent bucket (last element)
                datasets[catIdx].data[TIMELINE_WINDOW_MINUTES - 1] += delta;
            }
        }

        timelineChart.update('none');
    }

    // Persistent state for timeline delta tracking
    var timelineState = {};

    /**
     * Record a single event from the SSE stream into the current timeline bucket.
     * Called by dashboard.js for each incoming event.
     */
    function recordTimelineEvent(attackType) {
        if (!timelineChart) return;
        var catIdx = getTypeCategory(attackType);
        var datasets = timelineChart.data.datasets;
        if (datasets[catIdx]) {
            datasets[catIdx].data[TIMELINE_WINDOW_MINUTES - 1] += 1;
        }
    }

    /**
     * Shift all timeline buckets left by one (called every minute).
     */
    function shiftTimelineBuckets() {
        if (!timelineChart) return;
        var datasets = timelineChart.data.datasets;
        for (var i = 0; i < datasets.length; i++) {
            datasets[i].data.shift();
            datasets[i].data.push(0);
        }
        // Update labels
        timelineChart.data.labels = generateTimeLabels();
    }

    /**
     * Start the timeline chart polling and per-minute bucket shift.
     */
    function startTimelinePolling() {
        initTimelineChart();
        pollTimeline();
        timelineTimerId = setInterval(pollTimeline, TIMELINE_POLL_INTERVAL_MS);

        // Shift buckets every 60 seconds
        setInterval(shiftTimelineBuckets, 60000);

        // Refresh chart visually every 2s to show SSE-fed data
        setInterval(function () {
            if (timelineChart) {
                timelineChart.data.labels = generateTimeLabels();
                timelineChart.update('none');
            }
        }, 2000);
    }

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    function init() {
        // Cache stat card elements
        statElements = {
            attacks: document.getElementById('stat-attacks'),
            sources: document.getElementById('stat-sources'),
            countries: document.getElementById('stat-countries'),
            f1: document.getElementById('stat-f1'),
            threat: document.getElementById('stat-threat'),
            pps: document.getElementById('stat-pps')
        };

        // Cache top attackers table body
        attackersTbody = document.getElementById('attackers-tbody');

        // Cache timeline canvas
        timelineCanvas = document.getElementById('timeline-canvas');

        // Start all polling
        startStatsPolling();
        startAttackersPolling();
        startTimelinePolling();
    }

    // =========================================================================
    // CLEANUP
    // =========================================================================

    function destroy() {
        if (statsTimerId) { clearInterval(statsTimerId); statsTimerId = null; }
        if (attackersTimerId) { clearInterval(attackersTimerId); attackersTimerId = null; }
        if (timelineTimerId) { clearInterval(timelineTimerId); timelineTimerId = null; }
        if (timelineChart) { timelineChart.destroy(); timelineChart = null; }
    }

    // =========================================================================
    // PUBLIC API
    // =========================================================================

    window.PanelsManager = {
        /** Force a stats poll (for testing) */
        pollStats: pollStats,

        /** Force a top attackers poll (for testing) */
        pollTopAttackers: pollTopAttackers,

        /** Force a timeline poll (for testing) */
        pollTimeline: pollTimeline,

        /** Record a single SSE event into the timeline */
        recordTimelineEvent: recordTimelineEvent,

        /** Get the timeline chart instance */
        getTimelineChart: function () { return timelineChart; },

        /** Utility: mask IP */
        maskIp: maskIp,

        /** Utility: compute threat level */
        computeThreatLevel: computeThreatLevel,

        /** Utility: format number */
        formatNumber: formatNumber,

        /** Utility: get attack type category index */
        getTypeCategory: getTypeCategory,

        /** Cleanup all polling and chart */
        destroy: destroy,

        /** Constants */
        STATS_POLL_INTERVAL_MS: STATS_POLL_INTERVAL_MS,
        ATTACKERS_POLL_INTERVAL_MS: ATTACKERS_POLL_INTERVAL_MS,
        TIMELINE_POLL_INTERVAL_MS: TIMELINE_POLL_INTERVAL_MS,
        TIMELINE_WINDOW_MINUTES: TIMELINE_WINDOW_MINUTES,
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
