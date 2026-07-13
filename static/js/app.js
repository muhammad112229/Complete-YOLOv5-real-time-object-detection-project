document.querySelectorAll(".upload-form").forEach((form) => {
  const dropZone = form.querySelector(".drop-zone");
  const input = form.querySelector("input[type='file']");
  const label = dropZone ? dropZone.querySelector("span") : null;
  const loading = form.querySelector(".loading-indicator");

  if (dropZone && input) {
    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("drag-over");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove("drag-over");
      });
    });

    dropZone.addEventListener("drop", (event) => {
      if (event.dataTransfer.files.length > 0) {
        input.files = event.dataTransfer.files;
        if (label) {
          label.textContent = event.dataTransfer.files[0].name;
        }
      }
    });

    input.addEventListener("change", () => {
      if (label && input.files.length > 0) {
        label.textContent = input.files[0].name;
      }
    });
  }

  form.addEventListener("submit", () => {
    if (loading) {
      loading.hidden = false;
    }
    form.querySelectorAll("button").forEach((button) => {
      button.disabled = true;
    });
  });
});
