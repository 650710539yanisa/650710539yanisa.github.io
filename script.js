const dropArea = document.querySelector(".upload-area");
const fileInput = document.getElementById("img");
const resultBox = document.getElementById("result");

// ===== DRAG EVENTS =====
dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("drag-over");
});

dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("drag-over");
});

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove("drag-over");

    const file = e.dataTransfer.files[0];
    if (!file) return;

    fileInput.files = e.dataTransfer.files;  // สำคัญ!
    previewFile(file);

    resultBox.innerHTML = 'Processing not started — Click “Predict”.';
});

// ===== CLICK TO OPEN FILE CHOOSER =====
dropArea.addEventListener("click", () => fileInput.click());

// ===== PREVIEW IMAGE =====
fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
        previewFile(file);

        // ✅ เคลียร์ผลลัพธ์เก่าเมื่อเลือกรูปใหม่
        resultBox.innerHTML = 'Processing not started — Click “Predict”.';
    }
});

function previewFile(file) {
    const img = document.getElementById("preview");
    const container = document.getElementById("preview-container");

    img.src = URL.createObjectURL(file);
    container.style.display = "block";
}

// ===== CLEAR IMAGE =====
function clearImage() {
    fileInput.value = "";
    document.getElementById("preview-container").style.display = "none";
    document.getElementById("preview").src = "";
    document.getElementById("result").innerHTML =
        'No results yet — Upload an image and click “Predict”';
}

// ===== SEND IMAGE =====
async function sendImage() {
    const file = fileInput.files[0];
    if (!file) {
        alert("Please upload an image first");
        return;
    }

    let formData = new FormData();
    formData.append("image", file);

    // ✅ แสดงสถานะกำลังประมวลผล
    resultBox.innerHTML = "Running prediction…";

    try {
        let res = await fetch("/predict", { method: "POST", body: formData });
        let data = await res.json();

        // ✅ ใช้ result_html ที่ Flask ส่งกลับมา
        resultBox.innerHTML = data.result_html;
    } catch (err) {
        console.error(err);
        resultBox.innerHTML = "Error while predicting. Please try again.";
    }
}

// ✅ ให้ปุ่มใน HTML เรียกใช้ฟังก์ชันได้
window.sendImage = sendImage;
window.clearImage = clearImage;