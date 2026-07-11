(function () {
  var tabs = document.querySelectorAll(".tab[data-filter]");
  var list = document.getElementById("post-list");
  if (!tabs.length || !list) return;

  var items = list.querySelectorAll(".post-list__item");

  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      var filter = tab.getAttribute("data-filter");

      tabs.forEach(function (t) {
        t.setAttribute("aria-selected", String(t === tab));
      });

      items.forEach(function (item) {
        var cats = (item.getAttribute("data-categories") || "").split(" ");
        var show = filter === "all" || cats.indexOf(filter) !== -1;
        item.style.display = show ? "" : "none";
      });
    });
  });
})();
