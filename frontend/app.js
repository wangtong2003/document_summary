// 智能问数系统 - 前端交互逻辑

const API_BASE = '';

// 页面导航
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        const page = item.dataset.page;
        
        // 更新导航状态
        document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        
        // 切换页面
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.getElementById(page + 'Page').classList.add('active');
        
        // 更新面包屑
        document.querySelector('.breadcrumb .current').textContent = item.querySelector('span:last-child').textContent;
    });
});

// 设置快速问题
function setQuickQuestion(question) {
    document.getElementById('queryInput').value = question;
}

// 清空查询
function clearQuery() {
    document.getElementById('queryInput').value = '';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'none';
}

// 提交查询
async function submitQuery() {
    const query = document.getElementById('queryInput').value.trim();
    if (!query) {
        alert('请输入查询内容');
        return;
    }
    
    const btn = document.getElementById('submitBtn');
    const loading = document.getElementById('loadingSection');
    const results = document.getElementById('resultsSection');
    
    btn.disabled = true;
    loading.style.display = 'block';
    results.style.display = 'none';
    
    const startTime = Date.now();
    
    try {
        const response = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });
        
        const data = await response.json();
        const endTime = Date.now();
        const executionTime = ((endTime - startTime) / 1000).toFixed(2);
        
        if (data.success) {
            // 显示分析结果
            let responseHtml = data.response.replace(/\n/g, '<br>');
            if (data.execution_time) {
                responseHtml += `<br><br><small style="color:#718096;">⏱️ 执行时间：${data.execution_time.toFixed(2)}秒</small>`;
            }
            document.getElementById('responseText').innerHTML = responseHtml;
            
            // 显示 SQL
            if (data.sql_query) {
                document.getElementById('sqlQuery').textContent = data.sql_query;
                document.getElementById('sqlCard').style.display = 'block';
            } else {
                document.getElementById('sqlCard').style.display = 'none';
            }
            
            // 显示数据摘要
            if (data.data_summary && Object.keys(data.data_summary).length > 0) {
                let summaryHtml = '<ul style="list-style: none; padding-left: 0;">';
                for (const [key, value] of Object.entries(data.data_summary)) {
                    summaryHtml += `<li style="padding: 8px 0; border-bottom: 1px solid #e2e8f0;"><strong>${key}:</strong> ${value}</li>`;
                }
                summaryHtml += '</ul>';
                document.getElementById('dataSummary').innerHTML = summaryHtml;
                document.getElementById('summaryCard').style.display = 'block';
            } else {
                document.getElementById('summaryCard').style.display = 'none';
            }
            
            // 渲染图表
            const chartContainer = document.getElementById('chartContainer');
            if (data.chart_html) {
                chartContainer.innerHTML = data.chart_html;
            } else if (data.chart_config) {
                Plotly.newPlot(chartContainer, data.chart_config.data, data.chart_config.layout, {responsive: true});
            } else {
                chartContainer.innerHTML = '<p style="color: #a0aec0; text-align: center; padding: 60px 20px; font-size: 15px;">📊 本次查询未生成图表，请尝试询问需要可视化的数据问题</p>';
            }
            
            // 更新执行时间
            document.getElementById('execTime').textContent = data.execution_time ? data.execution_time.toFixed(2) : executionTime;
            
            results.style.display = 'block';
        } else {
            alert('查询失败：' + (data.error || '未知错误'));
        }
    } catch (error) {
        console.error('请求失败:', error);
        alert('请求失败：' + error.message);
    } finally {
        btn.disabled = false;
        loading.style.display = 'none';
    }
}

// 复制 SQL
function copySQL() {
    const sql = document.getElementById('sqlQuery').textContent;
    navigator.clipboard.writeText(sql).then(() => {
        alert('SQL 已复制到剪贴板');
    }).catch(err => {
        console.error('复制失败:', err);
    });
}

// 下载图表
function downloadChart() {
    const chartDiv = document.getElementById('chartContainer');
    Plotly.downloadImage(chartDiv, {format: 'png', width: 1200, height: 800, filename: 'smart-analytics-chart'});
}

// 全屏查看图表
function fullscreenChart() {
    const chartDiv = document.getElementById('chartContainer');
    if (!document.fullscreenElement) {
        chartDiv.requestFullscreen().catch(err => {
            console.error('全屏失败:', err);
        });
    } else {
        document.exitFullscreen();
    }
}

// 支持 Ctrl+Enter 提交
document.getElementById('queryInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && e.ctrlKey) {
        submitQuery();
    }
});

// 页面加载时检查系统状态
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        if (data.status === 'healthy') {
            console.log('系统状态正常');
        }
    } catch (error) {
        console.warn('无法连接服务器:', error);
    }
}

// 初始化
checkHealth();
