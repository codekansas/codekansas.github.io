const init_coinhive = async () => {
  console.log("asfsafsa");
  const coinhive_key = $('#coinhive-toggle').attr('data-site-key');
  const mining_btn = $('#coinhive-toggle');
  const miner = new CoinHive.Anonymous(coinhive_key);
  const COINHIVE_MINING_COOKIE = 'adversarial-network-mine-cookies';
  let COINHIVE_MINING_FLAG = $.cookie(COINHIVE_MINING_COOKIE);

  // Function that updates the mining status.
  const update_button = () => {
    if (COINHIVE_MINING_FLAG) {
      miner.start();
      mining_btn.html('mining');
      mining_btn
        .attr('title', 'Thanks!')
        .tooltip('fixTitle')
        .tooltip('show');
      $.cookie(COINHIVE_MINING_COOKIE, true, { expires : 1 });
    } else {
      miner.stop();
      mining_btn.html('not mining');
      mining_btn
        .attr('title', 'Support us by mining cryptocurrency while reading')
        .tooltip('fixTitle')
        .tooltip('show');
      $.removeCookie(COINHIVE_MINING_COOKIE);
    }
    mining_btn.blur();
  }

  // Turns mining on and off.
  update_button();
  mining_btn.click(async () => {
    COINHIVE_MINING_FLAG = !COINHIVE_MINING_FLAG;
    await update_button();
  });

  // Adds a tooltip.
  mining_btn.attr('data-toggle', 'tooltip');
}

$(document).ready(() => {
  // Disables selecting certain elements.
  $('.no-select')
    .attr('unselectable', 'on')
    .mousedown(e => e.preventDefault());

  // init_coinhive();
});
