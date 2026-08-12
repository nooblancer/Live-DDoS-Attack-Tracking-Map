/**
 * Replay Controls, Model Performance Card, and Attack Type Legend.
 *
 * Handles:
 *  - Start/stop buttons wired to POST /api/replay/start and /api/replay/stop
 *  - Speed selector (1x, 10x, 100x, 1000x) passing speed_multiplier parameter
 *  - "DATASET REPLAY" and "LIVE" labels permanently visible with state updates
 *  - Model card showing XGBoost type, CIC-DDoS2019 dataset, F1/precision/recall metrics
 *  - Attack type legend with live counts and color indicators
 *
 * Requirements: 10.1, 10.2, 10.3, 10.4, 12.1, 12.2, 12.3, 12.4
 */
(function () {
    'use strict';

    // =========================================================================
    // CONFIGURATION
    // =========================================================================

    var MODEL_STATS_POLL_INTERVAL_MS = 2000;
    var ATTACK_TYPES_POLL_INTERVAL_MS = 5000;

    // Attack type color mapping (same as globe.js and dashboard.js)
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

    // All 12 attack types in display order
    var ATTACK_TYPES = [
        'SYN Flood', 'UDP Flood', 'DNS Amplification', 'HTTP Flood',
        'LDAP', 'NTP', 'MSSQL', 'NetBIOS',
        'SSDP', 'TFTP', 'UDPLag', 'WebDDoS'
    ];

    // =========================================================================
    // STATE
    // =========================================================================

    var replayRunning = false;
    var currentSpeed = 10;
    var modelStatsTimer = null;
    var attackTypesTimer = null;

    // DOM element references
    var startBtn = null;
    var stopBtn = null;
    var speedSelector = null;
    var replaySpeedEl = null;
    var replayStatusEl = null;
    var replayIndicatorEl = null;
    var modelF1El = null;
    var modelPrecisionEl = null;
    var modelRecallEl = null;
    var modelTypeEl = null;
    var legendContainer = null;

    // =========================================================================
    // REPLAY CONTROLS
    // =========================================================================

    /**
     * Start the replay engine via POST /api/replay/start.
     * Sends the currently selected speed_multiplier.
     */
    function startReplay() {
        var speed = parseInt(speedSelector.value, 10) || 10;

        fetch('/api/replay/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ speed_multiplier: speed })
        })
        .then(function (response) {
            if (!response.ok) {
                throw new Error('Replay start failed: ' + response.status);
            }
            return response.json();
        })
        .then(function (data) {
            replayRunning = true;
            currentSpeed = data.speed || speed;
            updateReplayUI();
        })
        .catch(function (err) {
            console.error('[controls] Failed to start replay:', err);
        });
    }

    /**
     * Stop the replay engine via POST /api/replay/stop.
     */
    function stopReplay() {
        fetch('/api/replay/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        })
        .then(function (response) {
            if (!response.ok) {
                throw new Error('Replay stop failed: ' + response.status);
            }
            return response.json();
        })
        .then(function () {
            replayRunning = false;
            updateReplayUI();
        })
        .catch(function (err) {
            console.error('[controls] Failed to stop replay:', err);
        });
    }

    /**
     * Handle speed selector change. If replay is running, restart with new speed.
     */
    function onSpeedChange() {
        currentSpeed = parseInt(speedSelector.value, 10) || 10;
        updateReplayUI();

        // If replay is currently running, restart with new speed
        if (replayRunning) {
            startReplay();
        }
    }

    /**
     * Update the replay indicator UI elements to reflect current state.
     */
    function updateReplayUI() {
        if (replaySpeedEl) {
            replaySpeedEl.textContent = currentSpeed + 'x';
        }
        if (replayStatusEl) {
            replayStatusEl.textContent = replayRunning ? 'RUNNING' : 'STOPPED';
            replayStatusEl.style.color = replayRunning
                ? 'var(--color-secondary)'
                : 'var(--text-muted)';
        }
        if (replayIndicatorEl) {
            replayIndicatorEl.style.borderColor = replayRunning
                ? 'var(--color-secondary)'
                : 'var(--color-secondary-dim)';
        }
    }

    // =========================================================================
    // MODEL PERFORMANCE CARD
    // =========================================================================

    /**
     * Fetch model stats from GET /api/model-stats and update the model card.
     */
    function fetchModelStats() {
        fetch('/api/model-stats')
            .then(function (response) {
                if (!response.ok) {
                    throw new Error('Model stats fetch failed: ' + response.status);
                }
                return response.json();
            })
            .then(function (data) {
                updateModelCard(data);
            })
            .catch(function (err) {
                console.error('[controls] Failed to fetch model stats:', err);
            });
    }

    /**
     * Update the model performance card with fresh data.
     *
     * @param {Object} data - ModelStatsResponse payload
     */
    function updateModelCard(data) {
        if (modelTypeEl && data.model_type) {
            modelTypeEl.textContent = data.model_type;
        }
        if (modelF1El && data.f1_score != null) {
            modelF1El.textContent = data.f1_score.toFixed(4);
        }
        if (modelPrecisionEl && data.precision != null) {
            modelPrecisionEl.textContent = data.precision.toFixed(4);
        }
        if (modelRecallEl && data.recall != null) {
            modelRecallEl.textContent = data.recall.toFixed(4);
        }
    }

    // =========================================================================
    // ATTACK TYPE LEGEND
    // =========================================================================

    /**
     * Initialize the attack type legend with all 12 types, colors, and zero counts.
     */
    function initLegend() {
        if (!legendContainer) return;

        legendContainer.innerHTML = '';

        for (var i = 0; i < ATTACK_TYPES.length; i++) {
            var type = ATTACK_TYPES[i];
            var color = ATTACK_TYPE_COLORS[type] || '#00FF41';

            var item = document.createElement('div');
            item.className = 'legend-item';
            item.setAttribute('data-type', type);

            item.innerHTML =
                '<span class="legend-dot" style="background:' + color + '"></span>' +
                '<span class="legend-name">' + type + '</span>' +
                '<span class="legend-count" id="legend-count-' + i + '">0</span>';

            legendContainer.appendChild(item);
        }
    }

    /**
     * Fetch attack type counts from GET /api/attack-types and update legend.
     */
    function fetchAttackTypes() {
        fetch('/api/attack-types')
            .then(function (response) {
                if (!response.ok) {
                    throw new Error('Attack types fetch failed: ' + response.status);
                }
                return response.json();
            })
            .then(function (data) {
                updateLegendCounts(data);
            })
            .catch(function (err) {
                console.error('[controls] Failed to fetch attack types:', err);
            });
    }

    /**
     * Update the legend count values from API response.
     *
     * @param {Object} data - dict of attack_type → count
     */
    function updateLegendCounts(data) {
        if (!legendContainer) return;

        for (var i = 0; i < ATTACK_TYPES.length; i++) {
            var type = ATTACK_TYPES[i];
            var countEl = document.getElementById('legend-count-' + i);
            if (countEl) {
                var count = data[type] || 0;
                countEl.textContent = count;
            }
        }
    }

    // =========================================================================
    // POLLING
    // =========================================================================

    /**
     * Start polling model stats and attack types at their respective intervals.
     */
    function startPolling() {
        // Initial fetch
        fetchModelStats();
        fetchAttackTypes();

        // Set up periodic polling
        modelStatsTimer = setInterval(fetchModelStats, MODEL_STATS_POLL_INTERVAL_MS);
        attackTypesTimer = setInterval(fetchAttackTypes, ATTACK_TYPES_POLL_INTERVAL_MS);
    }

    /**
     * Stop all polling timers.
     */
    function stopPolling() {
        if (modelStatsTimer) {
            clearInterval(modelStatsTimer);
            modelStatsTimer = null;
        }
        if (attackTypesTimer) {
            clearInterval(attackTypesTimer);
            attackTypesTimer = null;
        }
    }

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    function init() {
        // Cache DOM references
        startBtn = document.getElementById('replay-start-btn');
        stopBtn = document.getElementById('replay-stop-btn');
        speedSelector = document.getElementById('speed-selector');
        replaySpeedEl = document.getElementById('replay-speed');
        replayStatusEl = document.getElementById('replay-status');
        replayIndicatorEl = document.getElementById('replay-indicator');
        modelF1El = document.getElementById('model-f1');
        modelPrecisionEl = document.getElementById('model-precision');
        modelRecallEl = document.getElementById('model-recall');
        modelTypeEl = document.getElementById('model-type');
        legendContainer = document.getElementById('attack-type-legend');

        // Wire up replay control buttons
        if (startBtn) {
            startBtn.addEventListener('click', startReplay);
        }
        if (stopBtn) {
            stopBtn.addEventListener('click', stopReplay);
        }

        // Wire up speed selector change
        if (speedSelector) {
            speedSelector.addEventListener('change', onSpeedChange);
        }

        // Initialize legend with all 12 types
        initLegend();

        // Set initial replay UI state
        currentSpeed = speedSelector ? parseInt(speedSelector.value, 10) : 10;
        updateReplayUI();

        // Check initial replay status
        fetchReplayStatus();

        // Start polling for model stats and attack types
        startPolling();
    }

    /**
     * Fetch the current replay status on load to sync UI with backend state.
     */
    function fetchReplayStatus() {
        fetch('/api/replay/status')
            .then(function (response) {
                if (!response.ok) return null;
                return response.json();
            })
            .then(function (data) {
                if (!data) return;
                replayRunning = data.state === 'running';
                currentSpeed = data.speed_multiplier || 10;
                if (speedSelector) {
                    speedSelector.value = String(currentSpeed);
                }
                updateReplayUI();
            })
            .catch(function (err) {
                console.error('[controls] Failed to fetch replay status:', err);
            });
    }

    // =========================================================================
    // PUBLIC API
    // =========================================================================

    window.ControlsManager = {
        /** Start the replay (programmatic) */
        startReplay: startReplay,
        /** Stop the replay (programmatic) */
        stopReplay: stopReplay,
        /** Get current replay state */
        isReplayRunning: function () { return replayRunning; },
        /** Get current speed */
        getCurrentSpeed: function () { return currentSpeed; },
        /** Force refresh model stats */
        refreshModelStats: fetchModelStats,
        /** Force refresh attack type counts */
        refreshAttackTypes: fetchAttackTypes,
        /** Stop polling (for cleanup) */
        stopPolling: stopPolling,
        /** Constants */
        ATTACK_TYPE_COLORS: ATTACK_TYPE_COLORS,
        ATTACK_TYPES: ATTACK_TYPES
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
