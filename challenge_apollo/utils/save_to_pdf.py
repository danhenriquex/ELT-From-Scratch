from fpdf import FPDF


# Generate the comparison table
def save_table_to_txt(cosine_metrics, euclidean_metrics, k_values):
    with open("KNN_comparison.txt", "w") as file:
        file.write("KNN Classification Comparison (Cosine vs. Euclidean)\n")
        file.write(
            "k\tCosine Accuracy\tCosine AUC\tEuclidean Accuracy\tEuclidean AUC\n"
        )
        for k in k_values:
            file.write(
                f"{k}\t{cosine_metrics[k]['accuracy']:.4f}\t{cosine_metrics[k]['roc_auc']:.4f}\t"
                f"{euclidean_metrics[k]['accuracy']:.4f}\t{euclidean_metrics[k]['roc_auc']:.4f}\n"
            )


# Save the comparison table as PDF using FPDF in vertical format
def save_table_to_pdf(cosine_metrics, euclidean_metrics, k_values):
    """
    Save the comparison table as a PDF file using FPDF.

    Args:
        cosine_metrics (dict): The dictionary of cosine metrics.
        euclidean_metrics (dict): The dictionary of euclidean metrics.
        k_values (list): The list of k values to consider.

    Returns:
        None
    """

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(
        200,
        10,
        "KNN Classification Comparison (Cosine vs. Euclidean)",
        ln=True,
        align="C",
    )

    # Table headers (in vertical format)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(50, 10, "Metric", border=1)
    for k in k_values:
        pdf.cell(30, 10, f"k={k}", border=1)
    pdf.ln()

    # Add Cosine Accuracy row
    pdf.set_font("Arial", "", 10)
    pdf.cell(50, 10, "Cosine Accuracy", border=1)
    for k in k_values:
        pdf.cell(30, 10, f"{cosine_metrics[k]['accuracy']:.4f}", border=1)
    pdf.ln()

    # Add Cosine AUC row
    pdf.cell(50, 10, "Cosine AUC", border=1)
    for k in k_values:
        pdf.cell(30, 10, f"{cosine_metrics[k]['roc_auc']:.4f}", border=1)
    pdf.ln()

    # Add Euclidean Accuracy row
    pdf.cell(50, 10, "Euclidean Accuracy", border=1)
    for k in k_values:
        pdf.cell(30, 10, f"{euclidean_metrics[k]['accuracy']:.4f}", border=1)
    pdf.ln()

    # Add Euclidean AUC row
    pdf.cell(50, 10, "Euclidean AUC", border=1)
    for k in k_values:
        pdf.cell(30, 10, f"{euclidean_metrics[k]['roc_auc']:.4f}", border=1)
    pdf.ln()

    # Output the PDF file
    pdf.output("KNN_comparison.pdf")
