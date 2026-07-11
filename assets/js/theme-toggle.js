(function () {
  var root = document.documentElement;
  var btn = document.getElementById("theme-toggle");
  if (!btn) return;

  function sync() {
    var isDark = root.getAttribute("data-theme") === "dark";
    btn.textContent = isDark ? "☀" : "☾";
    btn.setAttribute("aria-label", "Switch to " + (isDark ? "light" : "dark") + " theme");
  }

  btn.addEventListener("click", function () {
    var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try {
      localStorage.setItem("theme", next);
    } catch (e) {}
    sync();
  });

  sync();
})();
