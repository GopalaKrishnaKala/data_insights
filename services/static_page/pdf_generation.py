from fpdf import FPDF
import pandas as pd
import io
import re
import streamlit as st

class StyledPDF(FPDF):
    def write_with_styling(self, text):
        """
        Write text to the PDF, handling **bold** styling.
        Args:
            text (str): The input text with **bold** markers.
        """
        bold_pattern = r'\*\*(.*?)\*\*'  # Matches text wrapped with **
        parts = re.split(bold_pattern, text)  # Split text into regular and bold parts
        
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Odd parts are bold
                self.set_font("Arial", size=12, style="B")
            else:  # Even parts are regular
                self.set_font("Arial", size=12)
            self.write(10, part)
            

    def add_table(self, df):
        """
        Adds a table to the PDF.
        
        Args:
            df (pd.DataFrame): The DataFrame to render as a table.
        """
        column_widths = [self.w / (len(df.columns) + 2)] * len(df.columns)
        spacing = 1.5
        
        # Add header row
        self.set_font("Arial", style="B", size=10)
        row_height = self.font_size
        self.set_fill_color(200, 200, 200)  # Light gray background for header
        for i, col in enumerate(df.columns):
            self.cell(column_widths[i], row_height * spacing, col, border=1, align="C", fill=True)
        self.ln()
        
        # Add table rows
        self.set_font("Arial", size=8)
        row_height = self.font_size
        self.set_fill_color(255, 255, 255)  # White background for rows
        for _, row in df.iterrows():
            for i, value in enumerate(row):
                self.cell(column_widths[i], row_height * spacing, str(value), border=1, align="C")
            self.ln()

    
    def add_large_table(self, df):
        """
        Adds a large table to the PDF, adjusting for large content dynamically, 
        and ensuring proper alignment, consistent widths, and wrapped text.
        
        Args:
            self (FPDF): The FPDF object.
            df (pd.DataFrame): The DataFrame containing the table content.
        """
        # Set a smaller font for large tables
        self.set_font("Arial", size=6)
        
        # Dynamically calculate column widths
        page_width = self.w - 2 * self.l_margin  # Total width available for content
        first_column_width = page_width * 0.2  # Allocate 20% of the width for the first column
        other_columns_width = (page_width - first_column_width) / (len(df.columns) - 1)  # Remaining columns
        column_widths = [first_column_width] + [other_columns_width] * (len(df.columns) - 1)
        row_height = self.font_size * 1.5  # Height multiplier for row spacing
    
        # Add header row
        self.set_fill_color(200, 200, 200)  # Light gray background for header
        
        # Step 1: Calculate the maximum height for the entire header row
        max_row_height = 0
        header_y = self.get_y()  # Starting Y position for the header row
        header_text_lines = []  # To store text wrapping lines for each header
        
        for i, col in enumerate(df.columns):
            x = self.get_x()
            y = self.get_y()
            # Calculate the height of the column header text
            lines = self.multi_cell(column_widths[i], row_height, str(col), border=0, split_only=True)
            header_text_lines.append(lines)  # Save wrapped lines for rendering later
            cell_height = len(lines) * row_height
            max_row_height = max(max_row_height, cell_height)  # Track the tallest cell height
            self.set_xy(x + column_widths[i], y)  # Move to the next cell without rendering
        
        # Step 2: Render the header row using the calculated maximum height
        self.set_xy(self.l_margin, header_y)  # Reset to the starting Y position of the row
        for i, col in enumerate(df.columns):
            x = self.get_x()
            self.multi_cell(column_widths[i], max_row_height / len(header_text_lines[i]), str(col), border=1, align="C", fill=True)
            self.set_xy(x + column_widths[i], header_y)  # Move horizontally to the next cell
        
        self.ln(max_row_height)  # Move the cursor to the next row
    
        # Add table rows with wrapping and height adjustment
        self.set_fill_color(255, 255, 255)  # White background for rows
        
        for _, row in df.iterrows():
            max_row_height = 0  # Reset max height for this row
            row_text_lines = []  # Store text wrapping lines for each cell in the row
            row_y = self.get_y()  # Starting Y position for this row
        
            # Step 1: Calculate the maximum height for the entire row
            for i, value in enumerate(row):
                if isinstance(value, (float, int)):
                    value = round(value, 2)
                cell_text = str(value) if not pd.isna(value) else "-"
                
                x = self.get_x()
                y = self.get_y()
                
                # Calculate the height of the cell text
                lines = self.multi_cell(column_widths[i], row_height, cell_text, border=0, split_only=True)
                row_text_lines.append(lines)  # Save wrapped lines for rendering later
                cell_height = len(lines) * row_height
                max_row_height = max(max_row_height, cell_height)  # Track the tallest cell height
                self.set_xy(x + column_widths[i], y)  # Move to the next cell without rendering
        
            # Step 2: Render the row using the calculated maximum height
            self.set_xy(self.l_margin, row_y)  # Reset to the starting Y position of the row
            for i, value in enumerate(row):
                if isinstance(value, (float, int)):
                    value = round(value, 2)
                cell_text = str(value) if not pd.isna(value) else "-"
                
                x = self.get_x()
                self.multi_cell(column_widths[i], max_row_height / len(row_text_lines[i]), cell_text, border=1, align="C", fill=True)
                self.set_xy(x + column_widths[i], row_y)  # Move horizontally to the next cell
        
            self.ln(max_row_height)  # Move the cursor to the next row


    
    def get_string_height(self, cell_width, text):
        """
        Calculate the height of the text if wrapped within the given cell width.
    
        Args:
            cell_width (float): The width of the cell.
            text (str): The text to calculate the height for.
    
        Returns:
            float: The calculated height of the text.
        """
        # Use a dummy multi_cell to calculate the height
        lines = self.multi_cell(cell_width, self.font_size * 1.5, text, border=0, split_only=True)
        return len(lines) * self.font_size * 1.5



def create_pdf_report(output_path, title, sections, plots_with_summary, freq_dict, summary_stat_df):
    """
    Creates a PDF report with the provided title and sections, applying bold formatting.
    
    Args:
        output_path (str): The file path to save the PDF.
        title (str): The title of the report.
        sections (dict): A dictionary with section titles as keys and their content as values.
    """
    pdf = StyledPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)

    # Add Title
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(10)  # Add some space below the title

    # Add Sections
    pdf.set_font("Arial", size=12)
    for section_title, section_content in sections.items():
        # Add section title
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(0, 10, f"{section_title}", ln=True)
        pdf.ln(5)

        # Add section content with styling
        pdf.set_font("Arial", size=12)
        pdf.write_with_styling(section_content + "\n")
        
        pdf.ln(10)  # Add a 10-point line space

    # Add Visuals
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 10, "Visuals", ln=True, align="C")
    pdf.ln(5)

    is_first_visual = True
    for vis in plots_with_summary:

        if not is_first_visual:
            pdf.add_page()
        
        fig = vis[0]
        summary = vis[1]
        plt_no = vis[2]
        page_width = pdf.w - 2 * pdf.l_margin  # Available page width
        # Add the image to the PDF
        current_y = pdf.get_y()  # Get the current cursor position

        # Set image dimensions in PDF
        img_width = pdf.w - 2 * pdf.l_margin  # Fit within the page width
        img_aspect_ratio = 600 / 800          # Height / Width of the image
        img_height = img_width * img_aspect_ratio
        
        pdf.image(f"exports/report_figures/plot_no_{plt_no}.png", x=pdf.l_margin, y=current_y, w=page_width)
        
        # Move the cursor below the image
        pdf.set_y(current_y + img_height + 5)  # Add 5 units of spacing below the image
        
        # Add the summary text below the image
        pdf.set_font("Arial", size=9)
        pdf.multi_cell(0, 5, summary.replace('\n', '').encode('latin-1', 'replace').decode('latin-1'))

        is_first_visual = False

        
    # Add Appendix
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 10, "Appendix", ln=True, align="C")
    pdf.ln(5)

    # Add Frequency Tables Sub-section
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "Frequency Tables", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "In this section, we will provide frequency tables for all the columns in the dataset. The frequency counts are going to be different for columns depending on its classification.")
    pdf.ln(5)

    for column_name, table in freq_dict.items():
        # Convert table to DataFrame if not already
        if not isinstance(table, pd.DataFrame):
            table = pd.DataFrame(table)
        
        # Calculate required space
        row_height = pdf.font_size * 1.5  # Adjust based on font size
        header_height = row_height
        rows_required_height = row_height * len(table) + header_height
        space_left_on_page = pdf.h - pdf.get_y() - pdf.b_margin
    
        # If not enough space, add a new page
        if rows_required_height > space_left_on_page:
            pdf.add_page()
    
        # Print the column name once before adding the table
        pdf.set_font("Arial", style="B", size=12)
        col_clean_name = column_name.replace('_', ' ').title()
        pdf.cell(0, 10, col_clean_name, ln=True)
    
        # Add the table
        pdf.add_table(table)
        pdf.ln(5)  # Add space after each table

    # Add summary statistics table
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "Summary Statistics", ln=True)
    pdf.add_large_table(summary_stat_df)
    pdf.ln(10)

    # Add correlation matrix heatmap
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "Correlation Matrix Heatmap", ln=True)
    
    page_width = pdf.w - 2 * pdf.l_margin  # Available page width
    # Ensure the image is added to the PDF
    pdf.image("exports/report_figures/heatmap_plot.png", x=pdf.l_margin, y=pdf.get_y(), w=page_width)

    # Save the PDF
    pdf.output(output_path)
    # print(f"PDF report saved to {output_path}")
