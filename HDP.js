document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("healthForm");
    const outputDiv = document.getElementById("output");

    form.addEventListener("submit", function (event) {
        event.preventDefault();
        const formData = new FormData(form);
        let outputHTML = "<h3>Submitted Details:</h3>";

        formData.forEach((value, key) => {
            outputHTML += `<p><strong>${key}:</strong> ${value ? value : "N/A"}</p>`;
        });

        outputDiv.innerHTML = outputHTML;
        outputDiv.style.display = "block"; // Show output box
    });
});
