/**
 * Enhanced Globe.gl visualization for the DDoS Threat Intelligence Center (v2).
 *
 * Features:
 *  - Color-coded arcs by attack type (12 types mapped to 6 color groups)
 *  - Impact ring animations at target coordinates on arc arrival
 *  - Hexagonal-bin heat overlay updated every 5 seconds
 *  - 200-arc FIFO cap with oldest eviction
 *  - Arc lifecycle: 2000ms flight + 500ms fade
 *  - Live vs replay arc differentiation (live = brighter, "LIVE" label)
 *  - pointsMerge(true) for static markers
 *
 * Exposes window.GlobeManager with addArc(event) for SSE consumer (task 13.2).
 *
 * Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 15.1, 15.4
 */
(function () {
    'use strict';

    // =========================================================================
    // ATTACK TYPE COLOR MAPPING
    // =========================================================================

    const ATTACK_TYPE_COLORS = {
        'SYN Flood': '#FF0040',        // red
        'UDP Flood': '#FF8C00',        // orange
        'DNS Amplification': '#9B59B6', // purple
        'HTTP Flood': '#FFD700',       // yellow
        'LDAP': '#00E5FF',             // cyan (reflection)
        'NTP': '#00E5FF',              // cyan (reflection)
        'MSSQL': '#00E5FF',            // cyan (reflection)
        'NetBIOS': '#00FF41',          // green (other)
        'SSDP': '#00FF41',             // green (other)
        'TFTP': '#00FF41',             // green (other)
        'UDPLag': '#00FF41',           // green (other)
        'WebDDoS': '#FFD700'           // yellow
    };

    const DEFAULT_COLOR = '#00FF41';

    // =========================================================================
    // CONFIGURATION
    // =========================================================================

    const MAX_ACTIVE_ARCS = 200;
    const ARC_FLIGHT_MS = 2000;
    const ARC_FADE_MS = 500;
    const ARC_TOTAL_LIFETIME_MS = ARC_FLIGHT_MS + ARC_FADE_MS;
    const HEX_UPDATE_INTERVAL_MS = 5000;
    const TARGET_LAT = 40.7128;
    const TARGET_LNG = -74.0060;
    const IDLE_TIMEOUT_MS = 10000;
    const AUTO_ROTATE_SPEED = 0.3;

    // =========================================================================
    // STATE
    // =========================================================================

    let globe = null;
    let activeArcs = [];
    let impactRings = [];
    let allAttackPoints = [];
    let idleTimer = null;
    let isAutoRotating = false;
    let hexUpdateTimer = null;

    // =========================================================================
    // DOM
    // =========================================================================

    let container = null;

    // =========================================================================
    // GLOBE INITIALIZATION
    // =========================================================================

    function initGlobe() {
        if (!container) {
            console.error('[globe] #globe-container not found');
            return;
        }

        if (typeof Globe === 'undefined') {
            console.error('[globe] Globe.gl not loaded - Globe is undefined');
            return;
        }

        // Use the globe panel's actual dimensions (not the container's min-height)
        var panel = container.closest('.globe-panel') || container;
        var width = panel.clientWidth || 600;
        var height = panel.clientHeight || 400;
        console.log('[globe] Initializing globe:', width, 'x', height);

        globe = Globe()
            .width(width)
            .height(height)
            .globeImageUrl('//unpkg.com/three-globe/example/img/earth-dark.jpg')
            .backgroundColor('#0a0f0a')
            .showAtmosphere(true)
            .atmosphereColor('#00FF41')
            .atmosphereAltitude(0.15)
            // --- Arcs layer (attack trajectories) ---
            .arcsData(activeArcs)
            .arcStartLat('startLat')
            .arcStartLng('startLng')
            .arcEndLat('endLat')
            .arcEndLng('endLng')
            .arcColor('color')
            .arcStroke(0.5)
            .arcDashLength(0.4)
            .arcDashGap(0.2)
            .arcDashAnimateTime(1500)
            .arcAltitudeAutoScale(0.3)
            .arcLabel(function (d) {
                return d.source_channel === 'live'
                    ? 'LIVE: ' + d.attack_type
                    : d.attack_type;
            })
            // --- Rings layer (impact effects at target) ---
            .ringsData(impactRings)
            .ringLat('lat')
            .ringLng('lng')
            .ringColor(function () {
                return function (t) {
                    return 'rgba(0, 255, 65, ' + (1 - t) + ')';
                };
            })
            .ringMaxRadius(3)
            .ringPropagationSpeed(2)
            .ringRepeatPeriod(0)
            // --- Hex-bin heat layer ---
            .hexBinPointsData(allAttackPoints)
            .hexBinPointLat('lat')
            .hexBinPointLng('lng')
            .hexBinPointWeight('weight')
            .hexBinResolution(3)
            .hexAltitude(function (d) { return Math.min(d.sumWeight * 0.006, 0.5); })
            .hexTopColor(function (d) { return hexWeightColor(d.sumWeight); })
            .hexSideColor(function (d) { return hexWeightColor(d.sumWeight); })
            // --- Performance ---
            .pointsMerge(true)
            (container);

        // Initial camera position — zoom out to fit panel
        globe.pointOfView({ lat: 30, lng: 0, altitude: 3.0 });

        // Handle window resize
        window.addEventListener('resize', function () {
            if (globe && container) {
                var p = container.closest('.globe-panel') || container;
                var w = p.clientWidth || 600;
                var h = p.clientHeight || 400;
                globe.width(w).height(h);
            }
        });

        // Auto-rotation
        startAutoRotation();
        container.addEventListener('mousedown', resetIdleTimer);
        container.addEventListener('wheel', resetIdleTimer);
        container.addEventListener('touchstart', resetIdleTimer);

        // Periodic hex-bin refresh every 5 seconds
        hexUpdateTimer = setInterval(refreshHexLayer, HEX_UPDATE_INTERVAL_MS);
    }

    // =========================================================================
    // HEX-BIN HEAT LAYER
    // =========================================================================

    function hexWeightColor(weight) {
        if (weight > 20) return '#FF0040';
        if (weight > 10) return '#FF8C00';
        if (weight > 5) return '#FFD700';
        if (weight > 2) return '#7FFF00';
        return '#00FF41';
    }

    function refreshHexLayer() {
        if (globe) {
            // Trigger re-render with current accumulated points
            globe.hexBinPointsData(allAttackPoints);
        }
    }

    // =========================================================================
    // ARC MANAGEMENT — 200-CAP FIFO
    // =========================================================================

    /**
     * Add an attack arc to the globe visualization.
     * This is the primary API called by the SSE consumer (task 13.2).
     *
     * @param {Object} event - Enhanced attack event from SSE
     * @param {number} event.latitude - Source latitude
     * @param {number} event.longitude - Source longitude
     * @param {string} event.attack_type - One of 12 attack types
     * @param {string} event.source_channel - "replay" or "live"
     * @param {string} [event.ip_address] - Source IP address
     * @param {number} [event.confidence] - Classification confidence 0-1
     * @param {string} [event.timestamp] - ISO 8601 timestamp
     */
    function addArc(event) {
        if (!globe) return;

        // FIFO eviction: remove oldest arc(s) if at capacity
        while (activeArcs.length >= MAX_ACTIVE_ARCS) {
            var evicted = activeArcs.shift();
            if (evicted._ringTimer) clearTimeout(evicted._ringTimer);
            if (evicted._removeTimer) clearTimeout(evicted._removeTimer);
        }

        // Determine arc color based on attack type and source channel
        var baseColor = ATTACK_TYPE_COLORS[event.attack_type] || DEFAULT_COLOR;
        var arcColor;

        if (event.source_channel === 'live') {
            // Live arcs: brighter — use full-opacity triple for glow effect
            arcColor = [baseColor, '#ffffff', baseColor];
        } else {
            // Replay arcs: standard brightness
            arcColor = baseColor;
        }

        var arc = {
            startLat: event.latitude,
            startLng: event.longitude,
            endLat: TARGET_LAT,
            endLng: TARGET_LNG,
            color: arcColor,
            attack_type: event.attack_type || 'UNKNOWN',
            source_channel: event.source_channel || 'replay',
            _createdAt: Date.now(),
            _ringTimer: null,
            _removeTimer: null
        };

        activeArcs.push(arc);
        globe.arcsData(activeArcs);

        // Schedule impact ring at target after flight completes
        arc._ringTimer = setTimeout(function () {
            addImpactRing(TARGET_LAT, TARGET_LNG);
        }, ARC_FLIGHT_MS);

        // Schedule arc removal after full lifecycle (flight + fade)
        arc._removeTimer = setTimeout(function () {
            removeArc(arc);
        }, ARC_TOTAL_LIFETIME_MS);

        // Accumulate point for hex-bin heat layer (cap at 2000 to prevent lag)
        allAttackPoints.push({
            lat: event.latitude,
            lng: event.longitude,
            weight: 1
        });
        if (allAttackPoints.length > 2000) {
            allAttackPoints.splice(0, allAttackPoints.length - 2000);
        }
    }

    /**
     * Remove a specific arc from the active set.
     */
    function removeArc(arc) {
        var idx = activeArcs.indexOf(arc);
        if (idx !== -1) {
            activeArcs.splice(idx, 1);
            if (globe) {
                globe.arcsData(activeArcs);
            }
        }
    }

    // =========================================================================
    // IMPACT RINGS
    // =========================================================================

    /**
     * Add an expanding impact ring at the specified coordinates.
     * Triggered when an arc's flight time elapses (arrives at target).
     */
    function addImpactRing(lat, lng) {
        var ring = {
            lat: lat,
            lng: lng,
            _createdAt: Date.now()
        };

        impactRings.push(ring);
        if (globe) {
            globe.ringsData(impactRings);
        }

        // Remove ring after propagation completes (~1500ms at speed 2, radius 3)
        setTimeout(function () {
            var idx = impactRings.indexOf(ring);
            if (idx !== -1) {
                impactRings.splice(idx, 1);
                if (globe) {
                    globe.ringsData(impactRings);
                }
            }
        }, 1500);
    }

    // =========================================================================
    // AUTO-ROTATION
    // =========================================================================

    function startAutoRotation() {
        isAutoRotating = true;
        requestAnimationFrame(rotateGlobe);
    }

    function stopAutoRotation() {
        isAutoRotating = false;
    }

    function rotateGlobe() {
        if (!isAutoRotating || !globe) return;
        var pov = globe.pointOfView();
        globe.pointOfView({
            lat: pov.lat,
            lng: pov.lng + AUTO_ROTATE_SPEED,
            altitude: pov.altitude
        });
        requestAnimationFrame(rotateGlobe);
    }

    function resetIdleTimer() {
        stopAutoRotation();
        clearTimeout(idleTimer);
        idleTimer = setTimeout(function () {
            startAutoRotation();
        }, IDLE_TIMEOUT_MS);
    }

    // =========================================================================
    // HEALTH CHECK
    // =========================================================================

    async function checkHealth() {
        try {
            var resp = await fetch('/health');
            if (!resp.ok) return;
            var health = await resp.json();
            var modelNotice = document.getElementById('model-notice');
            if (modelNotice && !health.model_loaded) {
                modelNotice.removeAttribute('hidden');
            }
        } catch (e) {
            console.error('[globe] Health check failed:', e);
        }
    }

    // =========================================================================
    // CLEANUP
    // =========================================================================

    function destroy() {
        if (hexUpdateTimer) {
            clearInterval(hexUpdateTimer);
            hexUpdateTimer = null;
        }
        clearTimeout(idleTimer);
        stopAutoRotation();

        // Clear all pending timers on arcs
        activeArcs.forEach(function (arc) {
            if (arc._ringTimer) clearTimeout(arc._ringTimer);
            if (arc._removeTimer) clearTimeout(arc._removeTimer);
        });

        activeArcs = [];
        impactRings = [];
        allAttackPoints = [];
    }

    // =========================================================================
    // PUBLIC API (exposed on window for SSE consumer and other modules)
    // =========================================================================

    window.GlobeManager = {
        /** Add an attack arc — primary interface for SSE consumer */
        addArc: addArc,

        /** Get current count of active arcs */
        getActiveArcCount: function () { return activeArcs.length; },

        /** Get total accumulated heat points */
        getHeatPointCount: function () { return allAttackPoints.length; },

        /** Tear down the globe and clear all timers */
        destroy: destroy,

        /** Constants exposed for other modules */
        ATTACK_TYPE_COLORS: ATTACK_TYPE_COLORS,
        MAX_ACTIVE_ARCS: MAX_ACTIVE_ARCS,
        TARGET_LAT: TARGET_LAT,
        TARGET_LNG: TARGET_LNG
    };

    // =========================================================================
    // BOOTSTRAP
    // =========================================================================

    function init() {
        container = document.getElementById('globe-container');
        initGlobe();
        checkHealth();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
