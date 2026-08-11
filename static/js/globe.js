/**
 * Globe.gl initialization and SSE client for the DDoS Attack Tracking Map.
 * Renders attack source markers, animated arcs, live statistics, and timeline chart.
 */
(function () {
    'use strict';

    // --- Configuration ---
    const ACCENT = '#00FF41';
    const ACCENT_DIM = 'rgba(0, 255, 65, 0.6)';
    const TARGET_LAT = 40.7128;   // Default target (NYC)
    const TARGET_LNG = -74.0060;
    const ARC_LIFETIME_MS = 4000;
    const IDLE_TIMEOUT_MS = 10000;
    const AUTO_ROTATE_SPEED = 0.3; // degrees per frame

    // --- State ---
    let pointData = [];
    let arcData = [];
    let globe = null;
    let idleTimer = null;
    let isAutoRotating = false;

    // --- DOM references ---
    const container = document.getElementById('globe-container');
    const modelNotice = document.getElementById('model-notice');
    const totalIpsEl = document.getElementById('total-ips');
    const countriesEl = document.getElementById('countries');
    const attacksLastHourEl = document.getElementById('attacks-last-hour');
    const latestTimestampEl = document.getElementById('latest-timestamp');
    const timelineBarsEl = document.querySelector('#timeline-chart .timeline-bars');
    const feedListEl = document.querySelector('#event-feed .feed-list');

    // --- Globe.gl Initialization ---
    function initGlobe() {
        globe = Globe()
            .globeImageUrl('//unpkg.com/three-globe/example/img/earth-night.jpg')
            .backgroundColor('#000000')
            .showAtmosphere(true)
            .atmosphereColor(ACCENT_DIM)
            .atmosphereAltitude(0.15)
            // Point layer (attack sources)
            .pointsData(pointData)
            .pointLat('lat')
            .pointLng('lng')
            .pointColor(() => ACCENT)
            .pointAltitude(0.01)
            .pointRadius('size')
            .pointsMerge(false)
            // Arc layer (attack animations)
            .arcsData(arcData)
            .arcStartLat('startLat')
            .arcStartLng('startLng')
            .arcEndLat('endLat')
            .arcEndLng('endLng')
            .arcColor('color')
            .arcDashLength(0.4)
            .arcDashGap(0.2)
            .arcDashAnimateTime(1500)
            .arcStroke(0.5)
            // Label/tooltip on click
            .onPointClick(handlePointClick)
            (container);

        // Set initial camera position
        globe.pointOfView({ lat: 30, lng: 0, altitude: 2.5 });

        // Start auto-rotation
        startAutoRotation();

        // Listen for user interaction to pause auto-rotation
        container.addEventListener('mousedown', resetIdleTimer);
        container.addEventListener('wheel', resetIdleTimer);
        container.addEventListener('touchstart', resetIdleTimer);
    }

    // --- Auto-Rotation ---
    function startAutoRotation() {
        isAutoRotating = true;
        requestAnimationFrame(rotateGlobe);
    }

    function stopAutoRotation() {
        isAutoRotating = false;
    }

    function rotateGlobe() {
        if (!isAutoRotating || !globe) return;
        const pov = globe.pointOfView();
        globe.pointOfView({ lat: pov.lat, lng: pov.lng + AUTO_ROTATE_SPEED, altitude: pov.altitude });
        requestAnimationFrame(rotateGlobe);
    }

    function resetIdleTimer() {
        stopAutoRotation();
        clearTimeout(idleTimer);
        idleTimer = setTimeout(function () {
            startAutoRotation();
        }, IDLE_TIMEOUT_MS);
    }

    // --- Tooltip ---
    let tooltipEl = null;

    function handlePointClick(point, event) {
        // Remove existing tooltip
        removeTooltip();

        if (!point) return;

        tooltipEl = document.createElement('div');
        tooltipEl.className = 'globe-tooltip';
        tooltipEl.innerHTML =
            '<strong>IP:</strong> ' + escapeHtml(point.ip) + '<br>' +
            '<strong>Country:</strong> ' + escapeHtml(point.country || '—') + '<br>' +
            '<strong>City:</strong> ' + escapeHtml(point.city || '—') + '<br>' +
            '<strong>ISP:</strong> ' + escapeHtml(point.isp || '—') + '<br>' +
            '<strong>Last seen:</strong> ' + escapeHtml(point.lastSeen || '—');

        tooltipEl.style.position = 'fixed';
        tooltipEl.style.left = (event.clientX + 12) + 'px';
        tooltipEl.style.top = (event.clientY + 12) + 'px';
        tooltipEl.style.zIndex = '50';
        document.body.appendChild(tooltipEl);

        // Remove on next click anywhere
        setTimeout(function () {
            document.addEventListener('click', removeTooltip, { once: true });
        }, 50);
    }

    function removeTooltip() {
        if (tooltipEl && tooltipEl.parentNode) {
            tooltipEl.parentNode.removeChild(tooltipEl);
            tooltipEl = null;
        }
    }

    // --- Data Fetching ---
    async function fetchInitialAttacks() {
        try {
            const resp = await fetch('/api/attacks');
            if (!resp.ok) return;
            const attacks = await resp.json();
            attacks.forEach(function (attack) {
                addPoint(attack);
            });
            globe.pointsData(pointData);
        } catch (e) {
            console.error('[globe] Failed to fetch initial attacks:', e);
        }
    }

    async function fetchStats() {
        try {
            const resp = await fetch('/api/stats');
            if (!resp.ok) return;
            const stats = await resp.json();
            renderStats(stats);
        } catch (e) {
            console.error('[globe] Failed to fetch stats:', e);
        }
    }

    async function fetchTimeline() {
        try {
            const resp = await fetch('/api/timeline');
            if (!resp.ok) return;
            const timeline = await resp.json();
            renderTimeline(timeline);
        } catch (e) {
            console.error('[globe] Failed to fetch timeline:', e);
        }
    }

    async function checkHealth() {
        try {
            const resp = await fetch('/health');
            if (!resp.ok) return;
            const health = await resp.json();
            if (!health.model_loaded) {
                modelNotice.removeAttribute('hidden');
            }
        } catch (e) {
            console.error('[globe] Health check failed:', e);
        }
    }

    // --- SSE Connection ---
    function connectSSE() {
        const evtSource = new EventSource('/events');

        evtSource.addEventListener('attack', function (e) {
            try {
                const data = JSON.parse(e.data);
                handleNewAttack(data);
            } catch (err) {
                console.error('[globe] SSE parse error:', err);
            }
        });

        evtSource.onerror = function () {
            console.warn('[globe] SSE connection error, will auto-reconnect');
        };
    }

    // --- Event Handling ---
    function handleNewAttack(data) {
        // Add point marker
        addPoint(data);
        globe.pointsData(pointData);

        // Add arc with fade-out
        addArc(data);

        // Update feed
        addFeedItem(data);

        // Refresh stats and timeline
        fetchStats();
        fetchTimeline();
    }

    function addPoint(attack) {
        // Skip if already exists
        const existing = pointData.find(function (p) { return p.ip === attack.ip_address; });
        if (existing) {
            existing.lastSeen = attack.last_seen || attack.timestamp || existing.lastSeen;
            return;
        }

        pointData.push({
            lat: attack.latitude,
            lng: attack.longitude,
            ip: attack.ip_address,
            country: attack.country || null,
            city: attack.city || null,
            isp: attack.isp || null,
            lastSeen: attack.last_seen || attack.timestamp || '',
            size: 0.5
        });
    }

    function addArc(attack) {
        const arc = {
            startLat: attack.latitude,
            startLng: attack.longitude,
            endLat: TARGET_LAT,
            endLng: TARGET_LNG,
            color: ACCENT
        };
        arcData.push(arc);
        globe.arcsData(arcData);

        // Remove arc after animation completes (fade-out)
        setTimeout(function () {
            const idx = arcData.indexOf(arc);
            if (idx !== -1) {
                arcData.splice(idx, 1);
                globe.arcsData(arcData);
            }
        }, ARC_LIFETIME_MS);
    }

    // --- Render Functions ---
    function renderStats(stats) {
        totalIpsEl.textContent = stats.total_ips != null ? stats.total_ips : '0';
        countriesEl.textContent = stats.countries != null ? stats.countries : '0';
        attacksLastHourEl.textContent = stats.attacks_last_hour != null ? stats.attacks_last_hour : '0';
        latestTimestampEl.textContent = stats.latest_timestamp
            ? formatTimestamp(stats.latest_timestamp)
            : '—';
    }

    function renderTimeline(timeline) {
        // Clear existing bars
        timelineBarsEl.innerHTML = '';

        if (!timeline || timeline.length === 0) return;

        const maxCount = Math.max.apply(null, timeline.map(function (b) { return b.count; }));
        const barHeight = 80; // matches CSS .timeline-bars height

        timeline.forEach(function (bucket) {
            const bar = document.createElement('div');
            bar.className = 'bar';
            const height = maxCount > 0 ? (bucket.count / maxCount) * barHeight : 0;
            bar.style.height = Math.max(height, 2) + 'px';
            bar.title = bucket.hour + ': ' + bucket.count + ' attacks';
            timelineBarsEl.appendChild(bar);
        });
    }

    function addFeedItem(data) {
        const li = document.createElement('li');
        const ts = data.timestamp ? formatTimestamp(data.timestamp) : 'now';
        li.textContent = data.ip_address + ' — ' + (data.country || '??') + ' [' + ts + ']';

        // Insert at top
        if (feedListEl.firstChild) {
            feedListEl.insertBefore(li, feedListEl.firstChild);
        } else {
            feedListEl.appendChild(li);
        }

        // Keep feed to max 50 items
        while (feedListEl.children.length > 50) {
            feedListEl.removeChild(feedListEl.lastChild);
        }
    }

    // --- Utilities ---
    function formatTimestamp(isoStr) {
        try {
            const d = new Date(isoStr);
            return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        } catch (_) {
            return isoStr;
        }
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    // --- Bootstrap ---
    async function init() {
        initGlobe();
        await checkHealth();
        await fetchInitialAttacks();
        await fetchStats();
        await fetchTimeline();
        connectSSE();
    }

    // Wait for Globe.gl to be available
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
