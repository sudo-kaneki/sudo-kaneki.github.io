(function () {
  var link = document.getElementById("favicon");
  if (!link) return;

  // Blinking the favicon means swapping the link's href on a timer. Animating
  // the SVG itself (SMIL/CSS) does not work: Chrome rasterizes an SVG favicon
  // once and never repaints it.
  var FONT = "ui-monospace,SFMono-Regular,Menlo,monospace";
  var GREEN = "%235ee2a0"; // --accent
  var DARK = "%230b0d0c"; // --bg

  function tile(bg, fg) {
    return (
      "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>" +
      "<rect width='32' height='32' fill='" +
      bg +
      "'/>" +
      "<text x='16' y='22' font-family='" +
      FONT +
      "' font-size='15' font-weight='700' text-anchor='middle' fill='" +
      fg +
      "'>WB</text></svg>"
    );
  }

  var ON = tile(GREEN, DARK); // the resting state, matching the nav brand
  var OFF = tile(DARK, GREEN); // inverted — reads as a cursor blink

  // A blinking tab icon is motion. Honour the same preference the CSS does.
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    link.href = ON;
    return;
  }

  var lit = true;
  var timer = null;

  function tick() {
    lit = !lit;
    link.href = lit ? ON : OFF;
  }

  function start() {
    if (timer) return;
    timer = setInterval(tick, 600); // terminal cursor cadence
  }

  function stop() {
    clearInterval(timer);
    timer = null;
    lit = true;
    link.href = ON; // never leave the tab showing the inverted frame
  }

  // Don't burn a timer on a tab nobody is looking at.
  document.addEventListener("visibilitychange", function () {
    if (document.hidden) stop();
    else start();
  });

  if (!document.hidden) start();
})();
