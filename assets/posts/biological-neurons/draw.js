const margin = { top: 5, bottom: 20, left: 30, right: 20 };

class Plot {
  constructor(svg_selector, base_width, base_height, color) {
    this.svg = d3.select(svg_selector);
    this.width = base_width - margin.left - margin.right;
    this.height = base_height - margin.top - margin.bottom;
    this.g = this.svg.append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    this.x = d3.scaleLinear().range([0, this.width]);
    this.y = d3.scaleLinear().range([this.height, 0]);

    const xfunc = d => this.x(d[0]);
    const yfunc = d => this.y(d[1]);
    this.line = d3.line().x(xfunc).y(yfunc);
    this.color = color;
  }

  plot(data) {
    this.x.domain(d3.extent(data, function(d) { return d[0]; }));
    this.y.domain(d3.extent(data, function(d) { return d[1]; }));

    // Clears the element's contents.
    this.g.selectAll('*').remove();

    this.g.append("g")
      .attr("transform", "translate(0," + this.height + ")")
      .call(d3.axisBottom(this.x).ticks(5))
      .select(".domain")
      .remove();

    this.g.append("g")
      .call(d3.axisLeft(this.y).ticks(5))
      .select(".domain")
      .remove();

    // Draws the line.
    this.g
      .append("path")
      .datum(data)
      .attr("fill", "none")
      .attr("stroke", this.color)
      .attr("stroke-linejoin", "round")
      .attr("stroke-linecap", "round")
      .attr("stroke-width", 1)
      .attr("d", this.line);
  }
}
