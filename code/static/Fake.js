// === FILE SELECTION ===
const fileInput = document.getElementById("fileInput");
const selectBtn = document.getElementById("selectBtn");
const uploadBtn = document.getElementById("uploadBtn");
const fileNameText = document.getElementById("fileName");
const statusDiv = document.getElementById("status");

// Allowed file types
const allowedTypes = ["text/csv", 
                      "application/vnd.ms-excel", 
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"];

// Open file explorer when "Select File" button is clicked
selectBtn.addEventListener("click", () => {
  fileInput.click();
});

// Show selected file name and validate type
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    const file = fileInput.files[0];
    if (!allowedTypes.includes(file.type)) {
      fileNameText.textContent = "❌ Invalid file type!";
      fileNameText.style.color = "red";
      fileInput.value = ""; // clear invalid file
      return;
    }

    fileNameText.textContent = `Selected: ${file.name}`;
    fileNameText.style.color = "#00ffb3";
  } else {
    fileNameText.textContent = "No file selected";
    fileNameText.style.color = "#ccc";
  }
});

// === UPLOAD HANDLER ===
uploadBtn.addEventListener("click", async () => {
  if (!fileInput.files.length) {
    statusDiv.innerText = "⚠️ Please select a CSV or Excel file first.";
    statusDiv.style.color = "orange";
    return;
  }

  const file = fileInput.files[0];
  if (!allowedTypes.includes(file.type)) {
    statusDiv.innerText = "❌ Invalid file type!";
    statusDiv.style.color = "red";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  statusDiv.innerText = "⏳ Uploading... Please wait.";
  statusDiv.style.color = "#87cefa";

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) throw new Error("Upload failed.");

    const data = await response.json();
    statusDiv.innerText = data.message || "✅ File uploaded successfully!";
    statusDiv.style.color = "#00ffb3";
  } catch (error) {
    console.error("Upload error:", error);
    statusDiv.innerText = "❌ Upload failed. Please try again.";
    statusDiv.style.color = "red";
  }
});