document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("healthForm");

    form.addEventListener("submit", function (event) {
        event.preventDefault();
        const formData = new FormData(form);
        let queryParams = new URLSearchParams();

        formData.forEach((value, key) => {
            queryParams.append(key, value);
        });

        // Redirect to Streamlit app with form data
        window.location.href = `https://your-streamlit-app-url.streamlit.app/?${queryParams.toString()}`;
    });
});
