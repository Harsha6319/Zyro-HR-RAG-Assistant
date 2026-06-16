import os
import sys
from pathlib import Path

def install_and_import_reportlab():
    try:
        from reportlab.pdfgen import canvas
        return canvas
    except ImportError:
        print("Installing reportlab for mock PDF generation...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
        from reportlab.pdfgen import canvas
        return canvas

def create_policy_pdf(file_path: Path, title: str, sections: list) -> None:
    canvas_mod = install_and_import_reportlab()
    c = canvas_mod.Canvas(str(file_path))
    
    # Page 1: Title & Overview
    c.setFont("Helvetica-Bold", 20)
    c.drawString(100, 750, title)
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(100, 730, "Zyro Dynamics - Internal HR Policy Document")
    c.drawString(100, 715, "Confidential - For Internal Use Only")
    c.line(100, 705, 500, 705)
    
    y = 650
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, y, "1. Executive Summary")
    y -= 25
    c.setFont("Helvetica", 10)
    summary_text = f"This document outlines the official policies, rules, and guidelines for {title} at Zyro Dynamics. All employees are required to read, understand, and comply with the regulations detailed herein."
    c.drawString(100, y, summary_text[:90])
    c.drawString(100, y-15, summary_text[90:])
    y -= 60
    
    # Sections
    section_num = 2
    for sec_title, sec_text in sections:
        if y < 150:
            c.showPage()
            # Headers on new page
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(100, 750, f"Zyro Dynamics - {title}")
            c.line(100, 742, 500, 742)
            y = 700
            
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, y, f"{section_num}. {sec_title}")
        y -= 20
        c.setFont("Helvetica", 10)
        
        # Simple line wrap
        words = sec_text.split(" ")
        line = ""
        for word in words:
            if len(line + " " + word) > 85:
                c.drawString(100, y, line.strip())
                y -= 15
                line = word
                if y < 100:
                    c.showPage()
                    y = 700
            else:
                line += " " + word
        if line:
            c.drawString(100, y, line.strip())
            y -= 30
            
        section_num += 1
        
    c.showPage()
    c.save()
    print(f"Created: {file_path.name}")

def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 1. Company Profile
    create_policy_pdf(
        data_dir / "Company Profile.pdf",
        "Company Profile Policy",
        [
            ("Mission Statement", "Zyro Dynamics was founded in 2021 to build next-generation intelligent business solutions. Our mission is to accelerate human potential using smart workflows."),
            ("Founding and Core Values", "Zyro Dynamics values transparency, agility, innovation, and empathy. We are a remote-first organization with hubs across major cities.")
        ]
    )
    
    # 2. Employee Handbook
    create_policy_pdf(
        data_dir / "Employee Handbook.pdf",
        "Employee Handbook",
        [
            ("Core Working Hours", "The standard working hours at Zyro Dynamics are 10:00 AM to 6:00 PM local time. Flexible hours are permitted with manager approval."),
            ("Standards of Professional Conduct", "Employees are expected to act with high ethical standards, respect coworkers, and maintain professional behavior at all times.")
        ]
    )
    
    # 3. Leave Policy
    create_policy_pdf(
        data_dir / "Leave Policy.pdf",
        "Leave Policy",
        [
            ("Casual Leave", "All full-time employees are entitled to 12 days of Casual Leave (CL) per calendar year, credited pro-rata monthly."),
            ("Sick Leave", "Employees receive 10 days of Sick Leave (SL) annually. Medical certificates are required for sick leave exceeding three consecutive days."),
            ("Earned Leave", "Employees accrue 18 days of Earned Leave (EL) per year. EL can be carried forward up to a maximum of 30 days.")
        ]
    )
    
    # 4. Work From Home Policy
    create_policy_pdf(
        data_dir / "Work From Home Policy.pdf",
        "Work From Home Policy",
        [
            ("Eligibility", "All full-time desk-based employees are eligible for Work From Home (WFH) after completing their onboarding training period."),
            ("Home Office Stipend", "Zyro Dynamics provides a one-time remote workspace setup stipend of $500 to purchase ergonomic furniture and IT accessories."),
            ("Internet Reimbursement", "Active employees can claim monthly high-speed internet expenses up to $50 under remote working provisions.")
        ]
    )
    
    # 5. Code of Conduct
    create_policy_pdf(
        data_dir / "Code of Conduct.pdf",
        "Code of Conduct",
        [
            ("Conflict of Interest", "Employees must avoid any activity, investment, or association that interferes or appears to interfere with the interests of Zyro Dynamics."),
            ("Acceptable Use of Assets", "Company resources, including laptops, emails, and networks, must be used solely for business-related operations.")
        ]
    )
    
    # 6. Performance Review Policy
    create_policy_pdf(
        data_dir / "Performance Review Policy.pdf",
        "Performance Review Policy",
        [
            ("Cycle and Timeline", "Performance reviews are conducted bi-annually in June and December. Reviews consist of self-evaluation and manager assessment."),
            ("Rating Scale", "Ratings are based on a scale of 1 to 5: 1 (Unsatisfactory), 2 (Needs Improvement), 3 (Meets Expectations), 4 (Exceeds Expectations), 5 (Outstanding).")
        ]
    )
    
    # 7. Compensation & Benefits Policy
    create_policy_pdf(
        data_dir / "Compensation & Benefits Policy.pdf",
        "Compensation & Benefits Policy",
        [
            ("Salary Cycle", "Salaries are processed and paid on the last working day of each calendar month. Bonus payouts are processed in April."),
            ("Health Insurance Benefits", "Comprehensive health insurance coverage is provided to all full-time employees, including medical, dental, and vision support.")
        ]
    )
    
    # 8. IT & Data Security Policy
    create_policy_pdf(
        data_dir / "IT & Data Security Policy.pdf",
        "IT & Data Security Policy",
        [
            ("Device Security", "Laptops must have active antivirus software installed and firewall enabled. Users must lock screens when leaving their desks."),
            ("Password Complexity Requirements", "All system passwords must be at least 12 characters long, containing uppercase letters, numbers, and symbols.")
        ]
    )
    
    # 9. POSH Policy
    create_policy_pdf(
        data_dir / "POSH Policy.pdf",
        "POSH Policy (Prevention of Sexual Harassment)",
        [
            ("Zero Tolerance Directive", "Zyro Dynamics has a zero-tolerance policy for sexual harassment in any form at the workplace or during company events."),
            ("Internal Complaints Committee (ICC)", "Any employee experiencing harassment can submit a written complaint to the ICC at icc@zyrodynamics.com within three months of the incident.")
        ]
    )
    
    # 10. Onboarding & Separation Policy
    create_policy_pdf(
        data_dir / "Onboarding & Separation Policy.pdf",
        "Onboarding & Separation Policy",
        [
            ("New Hire Orientation", "New hires undergo a mandatory 3-day orientation covering company values, IT setups, and introductory team meetings."),
            ("Notice Period Regulations", "Permanent employees must serve a 60-day notice period upon resignation. Pro-rata salary adjustments apply during notice.")
        ]
    )
    
    # 11. Travel & Expense Policy
    create_policy_pdf(
        data_dir / "Travel & Expense Policy.pdf",
        "Travel & Expense Policy",
        [
            ("Travel Class Restrictions", "Domestic travel must be booked in Economy class. Business class is permitted only for international flights longer than 8 hours."),
            ("Meal Expense Daily Allowance", "Daily meal allowance is capped at $75 for domestic business trips. Itemized receipts are required for reimbursement claim approvals.")
        ]
    )
    
    print("All 11 policy PDFs have been successfully generated in 'data/'!")

if __name__ == "__main__":
    main()
