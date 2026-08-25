document.addEventListener('DOMContentLoaded', () => {
    // Initial fetches
    fetchStats();
    fetchNudge();

    // DOM Elements
    const chatInput = document.getElementById('chat-input');
    const chatSubmit = document.getElementById('chat-submit');
    const welcomeState = document.getElementById('welcome-state');
    const chatHistory = document.getElementById('chat-history');
    const timelineView = document.getElementById('timeline-view');
    const timelineContainer = document.getElementById('timeline-container');
    
    // Navigation items
    const navChat = document.querySelector('.nav-item.active'); // assumes first is chat
    const navTimeline = document.getElementById('nav-timeline');

    // State Tracking
    let hasChatted = false;

    // View Switcher
    function showChatView() {
        navTimeline.classList.remove('active');
        navChat.classList.add('active');
        timelineView.classList.add('hidden');
        
        if (hasChatted) {
            welcomeState.classList.add('hidden');
            chatHistory.classList.remove('hidden');
        } else {
            welcomeState.classList.remove('hidden');
            chatHistory.classList.add('hidden');
        }
    }

    function showTimelineView() {
        navChat.classList.remove('active');
        navTimeline.classList.add('active');
        welcomeState.classList.add('hidden');
        chatHistory.classList.add('hidden');
        timelineView.classList.remove('hidden');
        fetchSessions();
    }

    navChat.addEventListener('click', (e) => { e.preventDefault(); showChatView(); });
    navTimeline.addEventListener('click', (e) => { e.preventDefault(); showTimelineView(); });

    // Chat Submission
    chatSubmit.addEventListener('click', handleChatSubmit);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleChatSubmit();
    });

    async function handleChatSubmit() {
        const question = chatInput.value.trim();
        if (!question) return;

        // Transition from welcome state to chat history on first message
        if (!hasChatted) {
            hasChatted = true;
            welcomeState.classList.add('hidden');
            chatHistory.classList.remove('hidden');
        }

        appendMessage(question, 'user');
        chatInput.value = '';
        
        // Show typing indicator
        const loadingId = 'loading-' + Date.now();
        const loadingEl = document.createElement('div');
        loadingEl.id = loadingId;
        loadingEl.className = 'message ai';
        loadingEl.innerHTML = `
            <div class="msg-header"><i data-lucide="cpu" class="icon-sm"></i> Ambient Agent</div>
            <div class="msg-content" style="color: var(--text-muted);">Recalling context...</div>
        `;
        chatHistory.appendChild(loadingEl);
        lucide.createIcons({ root: loadingEl });
        chatHistory.parentElement.scrollTop = chatHistory.parentElement.scrollHeight;

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question})
            });
            const data = await res.json();
            
            document.getElementById(loadingId).remove();
            
            if (data.error) {
                appendMessage("Sorry, I encountered an error: " + data.error, 'ai');
            } else {
                appendMessage(data.answer, 'ai');
            }
        } catch (e) {
            document.getElementById(loadingId).remove();
            appendMessage("Network error querying agent.", 'ai');
        }
    }

    function appendMessage(text, sender) {
        const msg = document.createElement('div');
        msg.className = `message ${sender}`;
        
        if (sender === 'ai') {
            msg.innerHTML = `
                <div class="msg-header"><i data-lucide="cpu" class="icon-sm"></i> Ambient Agent</div>
                <div class="msg-content">${text}</div>
            `;
        } else {
            msg.innerHTML = `<div class="msg-content">${text}</div>`;
        }
        
        chatHistory.appendChild(msg);
        lucide.createIcons({ root: msg });
        chatHistory.parentElement.scrollTop = chatHistory.parentElement.scrollHeight;
    }

    // Quick Actions
    document.getElementById('btn-day-recap').addEventListener('click', () => {
        chatInput.value = "Give me a recap of my day based on my sessions.";
        handleChatSubmit();
    });

    // Data Fetchers
    async function fetchStats() {
        try {
            const res = await fetch('/api/stats');
            const stats = await res.json();
            const dbEl = document.getElementById('stat-db');
            const capsEl = document.getElementById('stat-caps');
            if (dbEl) dbEl.innerText = stats.db_size_mb;
            if (capsEl) capsEl.innerText = stats.total_captures;
        } catch (e) { console.error(e); }
    }

    async function fetchNudge() {
        const el = document.getElementById('nudge-text');
        if (!el) return;
        try {
            const res = await fetch('/api/nudge');
            const data = await res.json();
            el.innerText = data.nudge || "You're doing great. No new insights right now.";
        } catch (e) {
            el.innerText = "Could not fetch insights.";
        }
    }

    async function fetchSessions() {
        try {
            const res = await fetch('/api/sessions?limit=20');
            const sessions = await res.json();
            timelineContainer.innerHTML = '';
            
            if (sessions.length === 0) {
                timelineContainer.innerHTML = '<div style="color:var(--text-muted)">No activity found yet.</div>';
                return;
            }

            sessions.forEach(s => {
                const el = document.createElement('div');
                el.className = 'session-row';
                const st = new Date(s.start_time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                const et = new Date(s.end_time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                
                el.innerHTML = `
                    <div class="time">${st} - ${et}</div>
                    <div class="info">${s.label}</div>
                `;
                timelineContainer.appendChild(el);
            });
        } catch (e) { console.error(e); }
    }
});
