# services/interactive_page/create_pdf.py

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from textwrap import wrap


def create_pdf(buffer, responses_list, overall_summary):
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    left_margin = 50
    right_margin = width - 50
    top_margin = height - 50
    bottom_margin = 50

    y_position = top_margin

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(left_margin, y_position, "Data Insights Summary")
    y_position -= 30
    pdf.setFont("Helvetica", 12)

    pdf.line(left_margin, y_position + 10, right_margin, y_position + 10)

    for idx, response in enumerate(responses_list):
        question = response["question"]
        answer = response["answer"]
        image_path = response.get("image_path", None)

        if y_position < bottom_margin + 200:  # Add a new page if space runs out
            pdf.showPage()
            y_position = top_margin

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(left_margin, y_position, f"Q: {question}")
        y_position -= 15

        pdf.setFont("Helvetica", 12)
        wrapped_answer = wrap(answer, width=95)
        for line in wrapped_answer:
            if y_position < bottom_margin:
                pdf.showPage()
                y_position = top_margin
            pdf.drawString(left_margin, y_position, line)
            y_position -= 15

        if image_path:
            if y_position - 200 < bottom_margin:
                pdf.showPage()
                y_position = top_margin
            try:
                pdf.drawImage(image_path, left_margin, y_position - 200, width=400, height=200)
                y_position -= 220
            except Exception as e:
                pdf.drawString(left_margin, y_position, f"(Error displaying chart: {e})")
                y_position -= 15

        y_position -= 20  # Add spacing between Q&A pairs

    if y_position < bottom_margin:
        pdf.showPage()
        y_position = top_margin

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(left_margin, y_position, "Overall Summary:")
    y_position -= 20

    pdf.setFont("Helvetica", 12)
    wrapped_summary = wrap(overall_summary, width=95)
    for line in wrapped_summary:
        if y_position < bottom_margin:
            pdf.showPage()
            y_position = top_margin
        pdf.drawString(left_margin, y_position, line)
        y_position -= 15

    pdf.save()
