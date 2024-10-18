document.addEventListener('DOMContentLoaded', function() {
    // D3 트리 데이터를 서버에서 받아와서 그리기
    if (document.getElementById('tree-chart')) {
        fetch('http://localhost:8000/tree_data')
            .then(response => response.json())
            .then(treeData => {
                // D3 트리 그리기 로직
                function updateDimensions() {
                    // 브라우저 창 크기에 맞게 width와 height를 설정
                    var width = window.innerWidth,
                        height = window.innerHeight;
                    
                    return { width, height };
                }

                var { width, height } = updateDimensions();

                var margin = {top: 20, right: 90, bottom: 30, left: 90};
                width = width - margin.left - margin.right;
                height = height - margin.top - margin.bottom;

                var svg = d3.select("#tree-chart").append("svg")
                    .attr("width", width + margin.right + margin.left)
                    .attr("height", height + margin.top + margin.bottom)
                    .append("g")
                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


                var i = 0, duration = 750, root;
                var treemap = d3.tree().size([height, width]);
                root = d3.hierarchy(treeData, function(d) { return d.children; });
                root.x0 = height / 2;
                root.y0 = 0;

                root.children.forEach(collapse);
                update(root);

                function collapse(d) {
                    if (d.children) {
                        d._children = d.children;
                        d._children.forEach(collapse);
                        d.children = null;
                    }
                }

                function update(source) {
                    var treeData = treemap(root);
                    var nodes = treeData.descendants(), links = treeData.descendants().slice(1);
                    nodes.forEach(function(d){ d.y = d.depth * 250; });

                    var node = svg.selectAll('g.node').data(nodes, function(d) { return d.id || (d.id = ++i); });

                    var nodeEnter = node.enter().append('g').attr('class', 'node')
                        .attr("transform", function(d) { return "translate(" + source.y0 + "," + source.x0 + ")"; })
                        .on('click', click);

                    var rectHeight = 60, rectWidth = 200;
                    nodeEnter.append('rect')
                        .attr('class', 'node')
                        .attr("width", rectWidth)
                        .attr("height", rectHeight)
                        .attr("x", 0)
                        .attr("y", (rectHeight / 2) * -1)
                        .attr("rx", "5")
                        .style("fill", function(d) { return d.data.fill; });

                    nodeEnter.append('text')
                        .attr("dy", "-.35em")
                        .attr("x", 13)
                        .attr("text-anchor", "start")
                        .text(function(d) { return d.data.name; })
                        .append("tspan")
                        .attr("dy", "1.75em")
                        .attr("x", 13)
                        .text(function(d) { return d.data.subname; });

                    var nodeUpdate = nodeEnter.merge(node);
                    nodeUpdate.transition().duration(duration)
                        .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });

                    nodeUpdate.select('rect').style("fill", function(d) { return d.data.fill; }).attr('cursor', 'pointer');

                    var nodeExit = node.exit().transition().duration(duration)
                        .attr("transform", function(d) { return "translate(" + source.y + "," + source.x + ")"; })
                        .remove();

                    nodeExit.select('rect').attr('r', 1e-6);
                    nodeExit.select('text').style('fill-opacity', 1e-6);

                    var link = svg.selectAll('path.link').data(links, function(d) { return d.id; });

                    var linkEnter = link.enter().insert('path', "g").attr("class", "link")
                        .attr('d', function(d){
                            var o = {x: source.x0, y: source.y0};
                            return diagonal(o, o);
                        });

                    var linkUpdate = linkEnter.merge(link);
                    linkUpdate.transition().duration(duration)
                        .attr('d', function(d){ return diagonal(d, d.parent); });

                    var linkExit = link.exit().transition().duration(duration)
                        .attr('d', function(d) {
                            var o = {x: source.x, y: source.y};
                            return diagonal(o, o);
                        }).remove();

                    nodes.forEach(function(d){ d.x0 = d.x; d.y0 = d.y; });

                    function diagonal(s, d) {
                        path = `M ${s.y} ${s.x}
                                C ${(s.y + d.y) / 2} ${s.x},
                                  ${(s.y + d.y) / 2} ${d.x},
                                  ${d.y} ${d.x}`;
                        return path;
                    }

                    function click(d) {
                        if (d.children) {
                            d._children = d.children;
                            d.children = null;
                        } else {
                            d.children = d._children;
                            d._children = null;
                        }
                        update(d);
                    }
                }
            })
            .catch(error => console.error('Error fetching tree data:', error));
    }

    // Scatter Plot 데이터 가져와서 그리기
if (document.getElementById('scatter-plot')) {
        function drawScatterPlot() {
            var width = window.innerWidth * 0.8;  // 전체 화면의 80%로 설정
            var height = window.innerHeight * 0.7;  // 전체 화면의 70%로 설정
            
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
                        xaxis: { title: '시간' },
                        yaxis: { title: '긴장도' },
                        width: width,
                        height: height
                    };
                    Plotly.newPlot('scatter-plot', scatterData, scatterLayout);
                })
                .catch(error => console.error('Error fetching scatter data:', error));
        }

        // 처음 로드 시 차트 그리기
        drawScatterPlot();

        // 창 크기 변경 시 차트 크기 조정
        window.addEventListener('resize', function() {
            drawScatterPlot();
        });
    }

    // Radar Chart 데이터 가져와서 그리기
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
});
