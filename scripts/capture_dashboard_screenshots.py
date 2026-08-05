from pathlib import Path

from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_DIR = PROJECT_ROOT / "reports" / "screenshots"
DASHBOARD_URL = "http://localhost:8501"
CHROME_PATH = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
EDGE_PATH = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
SECTION_Y = {
    "Prediction": 114,
    "Model Performance": 137,
    "Threshold Tuning": 160,
    "Business Insights": 183,
}


def browser_path() -> str:
    if CHROME_PATH.exists():
        return str(CHROME_PATH)
    if EDGE_PATH.exists():
        return str(EDGE_PATH)
    raise FileNotFoundError("Chrome or Edge executable not found. Install a browser or adjust the path in this script.")


def capture(page, filename: str) -> None:
    page.wait_for_timeout(1000)
    page.screenshot(path=str(SCREENSHOT_DIR / filename), full_page=True)


def open_section(page, label: str, expected_text: str) -> None:
    page.mouse.click(70, SECTION_Y[label])
    page.wait_for_selector(f"text={expected_text}", timeout=30000)
    page.wait_for_timeout(1000)


def main() -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=browser_path(), headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.goto(DASHBOARD_URL, wait_until="domcontentloaded")
        page.wait_for_selector("text=Customer Churn Insights", timeout=30000)

        open_section(page, "Prediction", "Customer Risk Prediction")
        page.get_by_role("button", name="Predict churn risk").click()
        page.wait_for_selector("text=Churn probability", timeout=30000)
        capture(page, "dashboard_prediction.png")

        open_section(page, "Model Performance", "Model Comparison")
        capture(page, "dashboard_model_performance.png")

        open_section(page, "Threshold Tuning", "Best F1 threshold")
        capture(page, "dashboard_threshold_tuning.png")

        open_section(page, "Business Insights", "Retention Actions")
        capture(page, "dashboard_business_insights.png")

        browser.close()

    print(f"Saved screenshots to {SCREENSHOT_DIR}")


if __name__ == "__main__":
    main()