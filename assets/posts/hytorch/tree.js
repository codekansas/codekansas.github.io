const tree = (function() {
  function render_ast(container_ref, svg_width, svg_height, tree_data) {
    const margin = { top: 30, right: 50, bottom: 30, left: 50 },
      width = svg_width - margin.left - margin.right,
      height = svg_height - margin.top - margin.bottom;

    const svg = d3.select(container_ref).append('svg')
      .attr('width', svg_width)
      .attr('height', svg_height)
      .attr('viewBox', '0 0 ' + svg_width + ' ' + svg_height)
      .attr('class', 'ast-plot')
      .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    const duration = 250,
      radius = 10;

    function collapse(d) {
      if (d.children) {
        d._children = d.children;
        d._children.forEach(collapse);
        d.children = null;
      }
    }

    function get_max_depth(node) {
      if (!node.children) return 1;
      return 1 + Math.max(...node.children.map(get_max_depth));
    }

    const max_depth = get_max_depth(tree_data);

    const root_node = d3.hierarchy(tree_data, d => d.children);
    root_node.x0 = height / 2;
    root_node.y0 = 0;

    const treemap = d3.tree().size([height, width]);

    let i = 0;

    function diagonal(s, d) {
      path = `M ${s.y} ${s.x}
              C ${(s.y + d.y) / 2} ${s.x},
                ${(s.y + d.y) / 2} ${d.x},
                ${d.y} ${d.x}`
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

    function update(source) {
      const tree_data = treemap(root_node);

      const nodes = tree_data.descendants(),
        links = tree_data.descendants().slice(1);
      nodes.forEach(d => { d.y = d.depth * width / (max_depth - 1); });

      const node = svg.selectAll('g.node')
        .data(nodes, d => d.id || (d.id = ++i));

      const node_enter = node.enter().append('g')
        .attr('class', 'node')
        .attr('transform', d => 'translate(' + source.y0 + ',' + source.x0 + ')')
        .on('click', click);
      node_enter.append('circle')
        .attr('class', 'node')
        .attr('r', 1e-6)
        .style('fill', d => d._children ? 'lightsteelblue' : '#fff');
      node_enter.append('text')
        .attr('dy', '0.35em')
        .attr('x', radius + 3)
        .attr('y', radius + 3)
        .attr('text-anchor', 'start')
        .text(d => d.data.name);

      const node_update = node_enter.merge(node);
      node_update.transition()
        .duration(duration)
        .attr('transform', d => 'translate(' + d.y + ',' + d.x + ')');
      node_update.select('circle.node')
        .attr('r', radius)
        .style('fill', d => d._children ? 'lightsteelblue' : '#fff')
        .attr('cursor', 'pointer');

      const node_exit = node.exit().transition()
        .duration(duration)
        .attr('transform', d => 'translate(' + source.y + ',' + source.x + ')')
        .remove();
      node_exit.select('circle')
        .attr('r', 1e-6);
      node_exit.select('text')
        .style('fill-opacity', 1e-6);

      const link = svg.selectAll('path.link')
        .data(links, d => d.id);

      const link_enter = link.enter().insert('path', 'g')
        .attr('class', 'link')
        .attr('d', d => {
          const o = { x: source.x0, y: source.y0 };
          return diagonal(o, o);
        });

      const link_update = link_enter.merge(link);
      link_update.transition()
        .duration(duration)
        .attr('d', d => diagonal(d, d.parent));

      const link_exit = link.exit().transition()
        .duration(duration)
        .attr('d', d => {
          const o = { x: source.x, y: source.y };
          return diagonal(o, o);
        })
        .remove();
      nodes.forEach(d => { d.x0 = d.x; d.y0 = d.y; });
    }

    update(root_node);
  }

  return {
    render_ast: render_ast,
  }
})();
