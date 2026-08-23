window.easyicuDesktopStatus = function (kind, message) {
  const status = document.getElementById("status");
  const card = document.querySelector(".launch-card");
  if (!status || !card) return;
  status.textContent = String(message || "EasyICU 无法启动。");
  card.dataset.status = String(kind || "error");
};
