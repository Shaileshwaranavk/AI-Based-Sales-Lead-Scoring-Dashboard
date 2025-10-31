// ===== Global State =====
const uploadState = {
    newDataFile: null,
    historicalFile: null
};

// ===== File Handling =====
document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("newDataFile").addEventListener("change", (e) => {
        uploadState.newDataFile = e.target.files[0];
    });

    document.getElementById("labeledDataFile").addEventListener("change", (e) => {
        uploadState.historicalFile = e.target.files[0];
    });
});

// ===== Core Function: Generate Predictions =====
async function generatePredictions() {
    const productName = document.getElementById("productName").value.trim();
    const description = document.getElementById("productDescription").value.trim();
    const features = document.getElementById("productFeatures").value.trim();

    if (!productName || !uploadState.newDataFile) {
        showNotification("❌ Please provide Product Name and a New Leads CSV file.", "error");
        return;
    }

    // Prepare FormData
    const formData = new FormData();
    formData.append("product_name", productName);
    formData.append("description", description);
    formData.append("features", features);
    formData.append("new_data", uploadState.newDataFile);

    if (uploadState.historicalFile) {
        formData.append("labeled_data", uploadState.historicalFile);
    }

    toggleLoading(true);
    showNotification("⏳ Uploading and analyzing leads... Please wait.", "success");

    try {
        const response = await fetch("/api/analyze_leads/", {
            method: "POST",
            body: formData
        });

        toggleLoading(false);

        if (!response.ok) {
            throw new Error(`Server error (${response.status})`);
        }

        const result = await response.json();

        if (result.error) {
            showNotification(`❌ ${result.error}`, "error");
        } else {
            showNotification("✅ Predictions generated successfully!", "success");
            displayResults(result);
        }

    } catch (err) {
        toggleLoading(false);
        console.error(err);
        showNotification(`⚠️ Prediction failed: ${err.message}`, "error");
    }
}

// ===== Display Results =====
function displayResults(data) {
    const container = document.getElementById("resultsTable");
    if (!Array.isArray(data) || data.length === 0) {
        container.innerHTML = "<p class='text-gray-600'>No results returned from the server.</p>";
        return;
    }

    let tableHTML = `
        <table class="min-w-full border border-gray-300 text-sm">
            <thead class="bg-blue-50 text-blue-800">
                <tr>
                    <th class="border p-2">Lead ID</th>
                    <th class="border p-2">Hybrid Score</th>
                    <th class="border p-2">Explanation</th>
                </tr>
            </thead>
            <tbody>
    `;

    data.forEach((lead) => {
        tableHTML += `
            <tr class="hover:bg-gray-50">
                <td class="border p-2">${lead.Lead_ID || "-"}</td>
                <td class="border p-2 text-center">${lead.Hybrid_Score ? lead.Hybrid_Score.toFixed(2) : (lead.Predicted_Conversion_Score_ML || 0).toFixed(2)}</td>
                <td class="border p-2">${lead.Explanation || "N/A"}</td>
            </tr>
        `;
    });

    tableHTML += "</tbody></table>";
    container.innerHTML = tableHTML;
}

// ===== Notifications =====
function showNotification(message, type = "success") {
    const area = document.getElementById("notificationArea");
    const colorClass = type === "error" ? "error-message" : "success-message";

    area.innerHTML = `
        <div class="${colorClass} p-3 rounded-md text-center font-medium mb-3 fade-in">
            ${message}
        </div>
    `;

    setTimeout(() => {
        area.innerHTML = "";
    }, 4000);
}

// ===== Loading Spinner =====
function toggleLoading(isVisible) {
    const loader = document.getElementById("loadingSection");
    loader.classList.toggle("hidden", !isVisible);
}
