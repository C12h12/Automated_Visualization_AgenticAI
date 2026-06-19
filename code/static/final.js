document.addEventListener("DOMContentLoaded", () => {

    const uploadBtn = document.getElementById("uploadBtn");
    const fileInput = document.getElementById("datasetUpload");
    const statusDiv = document.getElementById("successMessage");

    const tableHead = document.getElementById("tableHead");
    const tableBody = document.getElementById("tableBody");
    const previewSection = document.getElementById("previewSection");

    const uploadArea = document.getElementById("uploadArea");

    uploadArea.addEventListener("click", (e) => {
        if (e.target.id === "uploadBtn") return;
        fileInput.click();
    });

    // ================= UPLOAD CLICK =================
    uploadBtn.addEventListener("click", async () => {

        if (!fileInput.files.length) {
            statusDiv.innerText = "⚠️ Please select a file first";
            statusDiv.style.color = "red";
            return;
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        statusDiv.innerText = "⏳ Uploading...";
        statusDiv.style.color = "blue";

        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            // ✅ STATUS MESSAGE
            statusDiv.innerText = data.message;
            statusDiv.style.color = "green";

            // ================= DATA PREVIEW =================
            tableHead.innerHTML = "";
            tableBody.innerHTML = "";

            let headRow = "<tr>";
            data.columns.forEach(col => headRow += `<th>${col}</th>`);
            headRow += "</tr>";
            tableHead.innerHTML = headRow;

            data.preview.forEach(row => {
                let tr = "<tr>";
                data.columns.forEach(col => {
                    tr += `<td>${row[col] ?? ""}</td>`;
                });
                tr += "</tr>";
                tableBody.innerHTML += tr;
            });

            previewSection.style.display = "block";

            // ================= KPI RENDER =================
            const kpiCardsContainer = document.getElementById("kpiCards");
            const kpiSection = document.getElementById("kpiSection");

            kpiCardsContainer.innerHTML = "";
            let count = 0;

            for (const [key, value] of Object.entries(data.kpis)) {
                if (count >= 3) break;
                const card = document.createElement("div");
                card.classList.add("kpi-card");
                card.innerHTML = `<h3>${key}</h3><p>${value}</p>`;
                kpiCardsContainer.appendChild(card);
                count++;
            }

            kpiSection.style.display = "block";

            // ================= DASHBOARD (CHART IMAGES) =================
            renderChartsAsImages(data.charts);

            previewSection.scrollIntoView({ behavior: "smooth" });

        } catch (err) {
            console.error(err);
            statusDiv.innerText = "❌ Upload failed";
            statusDiv.style.color = "red";
        }
    });

    // ================= CHART RENDER FUNCTION =================
    async function renderChartsAsImages(charts) {

        const dashboardSection = document.getElementById("dashboardSection");
        const chartsContainer = document.getElementById("chartsContainer");

        chartsContainer.innerHTML = "";

        for (const [name, chartJSON] of Object.entries(charts)) {

            const tempDiv = document.createElement("div");
            tempDiv.style.display = "none";
            document.body.appendChild(tempDiv);

            const fig = JSON.parse(chartJSON);

            await Plotly.newPlot(tempDiv, fig.data, fig.layout);

            const imgURL = await Plotly.toImage(tempDiv, {
                format: "png",
                width: 600,
                height: 400
            });

            const img = document.createElement("img");
            img.src = imgURL;
            img.alt = name;
            img.style.width = "100%";
            img.style.borderRadius = "12px";
            img.style.boxShadow = "0 6px 15px rgba(0,0,0,0.08)";

            chartsContainer.appendChild(img);
            document.body.removeChild(tempDiv);
        }

        dashboardSection.style.display = "block";
        dashboardSection.scrollIntoView({ behavior: "smooth" });
    }

});
