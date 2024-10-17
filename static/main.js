// Scatter Plot 데이터를 가져와서 그리기
if (document.getElementById('scatter-plot')) {
    fetch('http://localhost:8000/scatter_data')
        .then(response => response.json())
        .then(data => {
            var scatterData = [{
                x: data.x,
                y: data.y,
                mode: 'lines+markers',
                type: 'scatter',
                line: { shape: 'spline' },
            }];
            var scatterLayout = {
                title: '스토리 라인 그래프',
                xaxis: { title: '시간' },
                yaxis: { title: '긴장도' }
            };
            Plotly.newPlot('scatter-plot', scatterData, scatterLayout);
        })
        .catch(error => console.error('Error fetching scatter data:', error));
}

// Radar Chart 데이터를 가져와서 그리기
if (document.getElementById('radar-chart')) {
    fetch('http://localhost:8000/radar_data')
        .then(response => response.json())
        .then(data => {
            var radarData = [{
                type: 'scatterpolar',
                r: data.r,
                theta: data.theta,
                fill: 'toself',
                name: '캐릭터 성격'
            }];
            var radarLayout = {
                polar: {
                    radialaxis: {
                        visible: false,
                        range: [0, 5]
                    }
                },
                title: '캐릭터 성격'
            };
            Plotly.newPlot('radar-chart', radarData, radarLayout);
        })
        .catch(error => console.error('Error fetching radar data:', error));
}

