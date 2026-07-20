document.addEventListener('DOMContentLoaded', () => {
    const API_URL = '';
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const resultsList = document.getElementById('results-list');
    const resultsCount = document.getElementById('results-count');
    const serviceHealth = document.getElementById('service-health');
    const limitSlider = document.getElementById('limit');
    const limitVal = document.getElementById('limit-val');
    const useReranker = document.getElementById('use-reranker');
    const orgIdInput = document.getElementById('org-id');
    const orgPreset = document.getElementById('org-preset');
    const aclInput = document.getElementById('acl');
    const metricsSection = document.getElementById('metrics-section');
    const warningsContainer = document.getElementById('warnings-container');

    limitSlider.addEventListener('input', (e) => { limitVal.textContent = e.target.value; });
    orgPreset.addEventListener('change', () => {
        if (orgPreset.value !== 'custom') orgIdInput.value = orgPreset.value;
    });

    async function checkHealth() {
        try {
            const data = await (await fetch(`${API_URL}/health`)).json();
            const mv = data.features?.milvus ? 'MV✓' : 'MV✗';
            serviceHealth.textContent = `Online v${data.version} · ${mv} · dim ${data.embedding_dimension}`;
            serviceHealth.className = 'health-badge ok';
        } catch {
            serviceHealth.textContent = 'Offline';
            serviceHealth.className = 'health-badge error';
        }
    }
    checkHealth();
    setInterval(checkHealth, 30000);

    function escapeHtml(str) {
        return String(str || '')
            .replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    async function askAI(body, endpoint = '/v1/ask') {
        const section = document.getElementById('llm-section');
        const content = document.getElementById('llm-answer-content');
        section.style.display = 'block';
        content.textContent = 'Генерация…';
        try {
            const res = await fetch(`${API_URL}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Org-Id': body.org_id || '*',
                },
                body: JSON.stringify(body),
            });
            let raw = '';
            const reader = res.body.getReader();
            const dec = new TextDecoder();
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                raw += dec.decode(value, { stream: true });
                let answer = raw;
                if (answer.startsWith('__TRACE_ID__:')) {
                    answer = answer.split('\n\n').slice(1).join('\n\n');
                }
                if (answer) content.textContent = answer;
            }
        } catch (e) {
            content.textContent = `Ошибка: ${e.message}`;
        }
    }

    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) return;
        searchButton.disabled = true;
        searchButton.textContent = '…';
        resultsList.innerHTML = '<div class="empty-state"><p>Поиск в Milvus…</p></div>';
        const body = {
            query,
            limit: parseInt(limitSlider.value, 10),
            org_id: orgIdInput.value.trim() || '*',
            acl: aclInput.value.trim() ? aclInput.value.split(',').map((s) => s.trim()) : null,
            search_type: 'hybrid',
            use_reranker: useReranker.checked,
            rrf_k: 60,
        };
        try {
            const data = await (await fetch(`${API_URL}/v1/search/smart`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            })).json();

            if (data.needs_rag === false) {
                metricsSection.style.display = 'none';
                warningsContainer.style.display = 'none';
                resultsCount.textContent = `0 документов`;
                resultsList.innerHTML = '<div class="empty-state"><p>Поиск по базе знаний не требуется (общий вопрос).</p></div>';
                await askAI(body, '/v1/chat');
                return;
            }

            const m = data.metrics || {};
            metricsSection.style.display = 'grid';
            document.getElementById('metric-embed').textContent = `${m.embedding_time_ms ?? '—'} ms`;
            document.getElementById('metric-retrieve').textContent = `${m.retrieval_time_ms ?? '—'} ms`;
            document.getElementById('metric-rerank').textContent = `${m.rerank_time_ms ?? '—'} ms`;
            document.getElementById('metric-total').textContent = `${m.total_time_ms ?? '—'} ms`;

            const warnings = data.warnings || [];
            if (warnings.length) {
                warningsContainer.style.display = 'block';
                warningsContainer.innerHTML = warnings.map((w) => `<div class="warning-item">⚠ ${escapeHtml(w)}</div>`).join('');
            } else {
                warningsContainer.style.display = 'none';
            }

            const results = [...(data.results || [])].reverse();
            resultsCount.textContent = `${results.length} документов`;
            if (!results.length) {
                resultsList.innerHTML = '<div class="empty-state"><p>Ничего не найдено</p></div>';
            } else {
                resultsList.innerHTML = results.map((doc, idx) => {
                    const title = doc.metadata?.document_title || `#${idx + 1}`;
                    const score = doc.relevance_score ?? doc.score;
                    return `<div class="result-card glass" style="--card-index: ${idx}">
                        <div class="result-card-header">
                            <span class="result-title">${escapeHtml(title)}</span>
                            <span class="result-score-badge">${score != null ? Number(score).toFixed(4) : '—'}</span>
                        </div>
                        <div class="result-content">${escapeHtml(doc.content)}</div>
                    </div>`;
                }).join('');
            }
            await askAI(body, '/v1/ask');
        } catch (e) {
            resultsList.innerHTML = `<div class="empty-state"><p>Ошибка: ${escapeHtml(e.message)}</p></div>`;
        } finally {
            searchButton.disabled = false;
            searchButton.textContent = 'Найти';
        }
    }

    searchButton.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') performSearch(); });

    // Typical questions handler
    const questionPills = document.querySelectorAll('.question-pill');
    questionPills.forEach(pill => {
        pill.addEventListener('click', () => {
            searchInput.value = pill.textContent;
            performSearch();
        });
    });

    const modal = document.getElementById('admin-modal');
    document.getElementById('btn-admin-panel').onclick = () => { modal.style.display = 'flex'; };
    document.getElementById('btn-close-modal').onclick = () => { modal.style.display = 'none'; };

    document.getElementById('btn-submit-ingest').onclick = async () => {
        const title = document.getElementById('ingest-title').value.trim();
        const text = document.getElementById('ingest-text').value.trim();
        const orgId = document.getElementById('ingest-org-id').value.trim();
        const key = document.getElementById('admin-api-key').value.trim();
        if (!title || !text || !orgId) { alert('Заполните поля'); return; }
        const btn = document.getElementById('btn-submit-ingest');
        btn.disabled = true;
        try {
            const res = await fetch(`${API_URL}/v1/admin/ingest/text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-Admin-Key': key },
                body: JSON.stringify({ title, text, org_id: orgId, acl: ['public'], sync: true }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(JSON.stringify(data.detail || data));
            alert(`OK document_id=${data.document_id} chunks=${data.chunks_created}`);
            modal.style.display = 'none';
        } catch (e) {
            alert(e.message);
        } finally {
            btn.disabled = false;
        }
    };

    // Ambient blobs mouse-follow interactive effect
    document.addEventListener('mousemove', (e) => {
        const mouseX = (e.clientX / window.innerWidth - 0.5) * 50; // shift max 25px
        const mouseY = (e.clientY / window.innerHeight - 0.5) * 50;
        
        const blob1 = document.querySelector('.blob-1');
        const blob2 = document.querySelector('.blob-2');
        const blob3 = document.querySelector('.blob-3');
        
        if (blob1) blob1.style.transform = `translate(${mouseX}px, ${mouseY}px)`;
        if (blob2) blob2.style.transform = `translate(${-mouseX}px, ${-mouseY}px)`;
        if (blob3) blob3.style.transform = `translate(${mouseX * 0.5}px, ${-mouseY * 0.5}px)`;
    });
});
