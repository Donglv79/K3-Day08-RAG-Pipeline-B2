"""
Task 1 — Thu thập văn bản chính sách/quy định dịch vụ đại học.

Nguồn dữ liệu: trang công khai RMIT Vietnam (rmit.edu.vn) — đã xác minh HTTP 200
khi nghiên cứu. Gồm các văn bản về học phí, học bổng và nhà ở/ký túc xá,
đúng 4 chủ đề gợi ý trong LAB_GUIDE.

Các file tải về được lưu vào data/landing/legal/ với tên rõ ràng, không dấu.
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# (filename, url) — URL đã được verify HTTP 200
LEGAL_DOCS = [
    (
        "student-fees-and-charges-guide-2026.pdf",
        "https://www.rmit.edu.vn/assets/vn/en/assets-for-production/documents/pdfs/study-at-rmit/tuition-fees/student-fees-and-charges-guide-06-2026.pdf",
    ),
    (
        "rmit-university-vietnam-scholarship-terms-and-conditions.pdf",
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/documents/pdfs/study-at-rmit/scholarships/english-pdf/rmit-university-vietnam-scholarship-terms-and-conditions.pdf",
    ),
    (
        "accommodation-advice-for-international-students-in-vietnam.pdf",
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/documents/pdfs/students/accommodation/accommodation-advice-for-international-students-in-vietnam.pdf",
    ),
    (
        "hcm-accommodation-advice-list.pdf",
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/documents/pdfs/students/accommodation/hcm-accommodation-advice-list.pdf",
    ),
    (
        "hanoi-accommodation-advice-support-list.pdf",
        "https://www.rmit.edu.vn/content/dam/rmit/vn/en/assets-for-production/documents/pdfs/students/accommodation/hanoi-accommodation-advice-support-list.pdf",
    ),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str) -> Path:
    """
    Tải file PDF về DATA_DIR.

    Args:
        url: Direct link tới file PDF.
        filename: Tên file lưu trên đĩa.

    Returns:
        Đường dẫn file đã tải về.

    Raises:
        RuntimeError: nếu download thất bại hoặc file quá nhỏ (nghi lỗi).
    """
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()

    filepath = DATA_DIR / filename
    filepath.write_bytes(response.content)

    size = filepath.stat().st_size
    if size < 1024:
        filepath.unlink(missing_ok=True)
        raise RuntimeError(f"File {filename} quá nhỏ ({size} bytes), có thể là trang HTML lỗi")

    print(f"  ✓ Đã tải: {filepath.name} ({size:,} bytes)")
    return filepath


def download_all():
    """Tải toàn bộ văn bản trong LEGAL_DOCS."""
    setup_directory()
    downloaded, failed = [], []

    for filename, url in LEGAL_DOCS:
        print(f"Downloading: {filename}")
        try:
            download_file(url, filename)
            downloaded.append(filename)
        except Exception as e:
            print(f"  ✗ Lỗi {filename}: {e}")
            failed.append(filename)

    print(f"\n✓ Đã tải {len(downloaded)}/{len(LEGAL_DOCS)} file vào {DATA_DIR}")
    if failed:
        print(f"✗ Không tải được: {failed}")
    return downloaded, failed


if __name__ == "__main__":
    download_all()
