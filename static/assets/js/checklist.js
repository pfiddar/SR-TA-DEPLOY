document.addEventListener("DOMContentLoaded", () => {
  const relevantCheckboxes = document.querySelectorAll(
    'input[name="relevant_docs"]'
  );
  const irrelevantCheckboxes = document.querySelectorAll(
    'input[name="irrelevant_docs"]'
  );

  relevantCheckboxes.forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      const isChecked = checkbox.checked;
      if (isChecked) {
        // Uncheck all irrelevant checkboxes for this result
        const resultIndex = checkbox.value;
        const irrelevantCheckbox = document.querySelector(
          `input[name="irrelevant_docs"][value="${resultIndex}"]`
        );
        irrelevantCheckbox.checked = false;
      }
      validateForm();
    });
  });

  irrelevantCheckboxes.forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      const isChecked = checkbox.checked;
      if (isChecked) {
        // Uncheck all relevant checkboxes for this result
        const resultIndex = checkbox.value;
        const relevantCheckbox = document.querySelector(
          `input[name="relevant_docs"][value="${resultIndex}"]`
        );
        relevantCheckbox.checked = false;
      }
      validateForm();
    });
  });

  function validateForm() {
    let anyChecked = false;
    relevantCheckboxes.forEach((checkbox) => {
      if (checkbox.checked) {
        anyChecked = true;
      }
    });
    irrelevantCheckboxes.forEach((checkbox) => {
      if (checkbox.checked) {
        anyChecked = true;
      }
    });

    const submitButton = document.getElementById("submitFeedback");
    submitButton.style.display = anyChecked ? "block" : "none";
  }
});
