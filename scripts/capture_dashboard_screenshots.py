from pathlib import Path

from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_DIR = PROJECT_ROOT / "reports" / "screenshots"
DASHBOARD_URL = "http://localhost:8501"
CHROME_PATH = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
EDGE_PATH = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")


def browser_path() -> str:
    if CHROME_PATH.exists():
        return str(CHROME_PATH)
    if EDGE_PATH.exists():
        return str(EDGE_PATH)
    raise FileNotFoundError("Chrome or Edge executable not found. Install a browser or adjust the path in this script.")


def capture(page, filename: str) -> None:
    page.wait_for_timeout(1000)
    page.screenshot(path=str(SCREENSHOT_DIR / filename), full_page=True)


def click_tab(page, label: str) -> None:
    page.get_by_role("tab", name=label).click()
    page.wait_for_timeout(1500)


def main() -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=browser_path(), headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
        page.goto(DASHBOARD_URL, wait_until="networkidle")
        page.wait_for_selector("text=Customer Churn Insights", timeout=30000)

        page.get_by_role("button", name="Predict churn risk").click()
        capture(page, "dashboard_prediction.png")

        click_tab(page, "Model Performance")
        capture(page, "dashboard_model_performance.png")

        click_tab(page, "Threshold Tuning")
        capture(page, "dashboard_threshold_tuning.png")

        click_tab(page, "Business Insights")
        capture(page, "dashboard_business_insights.png")

        browser.close()

    print(f"Saved screenshots to {SCREENSHOT_DIR}")


if __name__ == "__main__":
    main()

