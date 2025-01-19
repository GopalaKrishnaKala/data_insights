import matplotlib.pyplot as plt
import uuid


def save_chart(response):

    if isinstance(response, plt.Figure):  # Case 1: Matplotlib figure
        chart_path = f"{uuid.uuid4().hex}.png"
        response.savefig(chart_path, bbox_inches='tight')
        plt.close(response)
        print(f"Chart saved successfully at: {chart_path}")
        return chart_path
    elif isinstance(response, str) and response.endswith(".png"):  # Case 2: Pre-saved chart path
        print(f"Existing chart detected at: {response}")
        return response
    else:
        print("No chart to save for this response.")
        return None
