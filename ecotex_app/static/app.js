(function () {
  const $ = (s) => document.querySelector(s);

  // ---- Dark mode toggle (works on every page) ----
  const root = document.documentElement;
  try { const t = localStorage.getItem("theme"); if (t) root.dataset.theme = t; } catch (e) {}
  $("#theme").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {}
  });

  if (!$("#go")) return; // rest is only for the Classify page

  let file = null, stream = null, last = null;
  const show = (id, on) => { $(id).hidden = !on; };

  function stopCamera() {
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    show("#video", false); show("#shot", false);
  }

  function setFile(f) {
    if (!f) return;
    stopCamera();
    file = f;
    $("#preview").src = URL.createObjectURL(f);
    show("#preview", true); show("#hint", false);
    $("#go").disabled = false;
    show("#res", false); show("#err", false); show("#empty", true);
  }

  $("#file").addEventListener("change", (e) => setFile(e.target.files[0]));
  $("#snap").addEventListener("change", (e) => setFile(e.target.files[0]));

  const drop = $("#drop");
  ["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  // ---- Live camera (needs HTTPS or localhost) ----
  $("#live").addEventListener("click", async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return showError("Live camera is not available here. Use 'Take photo' instead.");
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
      $("#video").srcObject = stream;
      show("#video", true); show("#shot", true); show("#preview", false); show("#hint", false);
    } catch (e) {
      showError("Camera permission denied or unavailable. Use 'Take photo' instead.");
    }
  });

  $("#shot").addEventListener("click", () => {
    const v = $("#video"), c = $("#canvas");
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext("2d").drawImage(v, 0, 0);
    c.toBlob((b) => setFile(new File([b], "capture.jpg", { type: "image/jpeg" })), "image/jpeg", 0.92);
  });

  function showError(msg) {
    show("#empty", false); show("#busy", false); show("#res", false);
    $("#err").textContent = msg; show("#err", true);
  }

  // ---- Classify ----
  $("#go").addEventListener("click", async () => {
    if (!file) return;
    show("#empty", false); show("#err", false); show("#res", false); show("#busy", true);
    $("#go").disabled = true;
    const fd = new FormData();
    fd.append("image", file);
    try {
      const r = await fetch("/predict", { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Something went wrong.");
      render(d);
    } catch (e) {
      showError(e.message === "Failed to fetch" ? "Cannot reach the server. Is it running?" : e.message);
    } finally {
      show("#busy", false);
      $("#go").disabled = false;
    }
  });

  function render(d) {
    last = d;
    $("#label").textContent = d.label + " (" + Math.round(d.confidence * 100) + "%)";
    show("#warn", d.uncertain);
    $("#bars").innerHTML = d.top3.map((t) => {
      const p = Math.round(t.confidence * 100);
      return '<div class="bar"><div class="top"><span>' + t.label + "</span><span>" + p +
        '%</span></div><div class="track"><div class="fill" data-w="' + p + '"></div></div></div>';
    }).join("");
    $("#tip").textContent = "Recycling tip: " + d.tip;
    show("#res", true);
    requestAnimationFrame(() => document.querySelectorAll(".fill").forEach((f) => { f.style.width = f.dataset.w + "%"; }));
  }

  // ---- Download result as a text file ----
  $("#save").addEventListener("click", () => {
    if (!last) return;
    const lines = ["EcoTex Sorter result", "Fabric: " + last.label,
      "Confidence: " + Math.round(last.confidence * 100) + "%",
      ...last.top3.map((t, i) => (i + 1) + ". " + t.label + " " + Math.round(t.confidence * 100) + "%"),
      "Tip: " + last.tip];
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/plain" }));
    a.download = "ecotex_result.txt";
    a.click();
  });
})();
