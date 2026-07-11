(function () {
  var root = document.documentElement;
  var btn = document.getElementById("theme-toggle");
  if (!btn) return;

  // The icon is rendered by CSS from [data-theme]; the aria-label is
  // state-independent. aria-pressed reflects whether dark (the default,
  // "pressed" state of this toggle) is currently active, so a
  // screen-reader user gets confirmation the toggle did something.
  btn.setAttribute("aria-pressed", String(root.getAttribute("data-theme") !== "light"));

  btn.addEventListener("click", function () {
    var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    btn.setAttribute("aria-pressed", String(next === "dark"));
    try {
      localStorage.setItem("theme", next);
    } catch (e) {}
  });
})();
