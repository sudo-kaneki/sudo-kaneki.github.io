(function () {
  var tabs = document.querySelectorAll(".tab[data-filter]");
  var list = document.getElementById("post-list");
  var empty = document.getElementById("post-empty");
  if (!tabs.length || !list) return;

  var items = list.querySelectorAll(".post-list__item");

  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      var filter = tab.getAttribute("data-filter");
      var shown = 0;

      tabs.forEach(function (t) {
        t.setAttribute("aria-pressed", String(t === tab));
      });

      items.forEach(function (item) {
        var cats = (item.getAttribute("data-categories") || "").split(",");
        var show = filter === "all" || cats.indexOf(filter) !== -1;
        item.hidden = !show;
        if (show) shown++;
      });

      if (empty) empty.hidden = shown !== 0;
    });
  });
})();
