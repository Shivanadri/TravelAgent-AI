import os
from datetime import datetime


def run_pdf_agent(state: dict) -> dict:
    from rich_ui import update_status

    update_status("pdf", "running")
    print("\n" + "=" * 55)
    print("   TRIP PLANNER v3 — Agent 11: PDF Agent")
    print("=" * 55)

    prefs     = state.get("trip_preferences", {})
    itinerary = state.get("itinerary", {})
    budget    = state.get("budget_summary", {})
    weather   = state.get("weather_data", {})
    transport = state.get("transport_data", {})
    hotel     = state.get("hotel_data", {})
    session   = state.get("session_id", "trip")

    os.makedirs("output", exist_ok=True)
    pdf_path     = f"output/trip_{session[:8]}.pdf"
    summary_path = f"output/whatsapp_{session[:8]}.txt"

    _write_whatsapp_summary(summary_path, prefs, itinerary, budget, weather, transport, hotel)

    pdf_success = _generate_pdf(pdf_path, prefs, itinerary, budget, weather, transport, hotel)

    if pdf_success:
        print(f"\n  ✓ PDF saved          : {pdf_path}")
    else:
        pdf_path = None
        print(f"\n  ⚠ PDF skipped (install reportlab to enable)")

    print(f"  ✓ WhatsApp summary   : {summary_path}")
    print("=" * 55)
    update_status("pdf", "done", "saved")

    return {
        "pdf_path":              pdf_path,
        "whatsapp_summary_path": summary_path,
        "last_completed_node":   "pdf_agent",
    }


def _write_whatsapp_summary(path, prefs, itinerary, budget, weather, transport, hotel):
    hotel_name  = hotel.get("recommended", {}).get("name", "TBD")
    hotel_rate  = hotel.get("price_per_night", 0)

    lines = [
        f"🧳 *{itinerary.get('title', 'Trip Plan')}*",
        f"📍 {prefs.get('source')} → {prefs.get('destination')}",
        f"📅 {prefs.get('start_date')} to {prefs.get('end_date')}",
        f"👥 {prefs.get('travelers')} traveler(s)",
        "",
        f"✈️ *Transport:* {transport.get('final_mode', '?').upper()} — Rs.{transport.get('total_transport_cost', 0):,}",
        f"🏨 *Hotel:* {hotel_name} — Rs.{hotel_rate:,}/night",
        "",
        f"🌤️ *Weather:* {weather.get('condition', '?')} (score {weather.get('score', '?')}/10)",
        "",
        f"💰 *Budget Summary:*",
        f"  Total: Rs.{budget.get('total_budget', 0):,}",
        f"  Estimate: Rs.{budget.get('total_estimate', 0):,}",
        f"  Status: {'✓ Within budget' if budget.get('within_budget') else '⚠ Over budget'}",
        "",
        f"📋 *Day-by-day:*",
    ]

    for day in itinerary.get("days", []):
        lines.append(f"  *Day {day.get('day')}* — {day.get('theme')}")
        for act in day.get("activities", [])[:3]:
            lines.append(f"    • {act.get('time', '')} {act.get('activity')} @ {act.get('location')}")

    tips = itinerary.get("travel_tips", [])
    if tips:
        lines += ["", "💡 *Travel Tips:*"]
        for tip in tips[:3]:
            lines.append(f"  • {tip}")

    text = "\n".join(lines)
    if len(text) > 3000:
        text = text[:2950] + "\n...(see full PDF)"

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _generate_pdf(path, prefs, itinerary, budget, weather, transport, hotel):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        return False

    doc    = SimpleDocTemplate(path, pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    title_s  = ParagraphStyle("title",  parent=styles["Title"],   fontSize=18, spaceAfter=6)
    h2_s     = ParagraphStyle("h2",     parent=styles["Heading2"], fontSize=13, spaceAfter=4)
    footer_s = ParagraphStyle("footer", parent=styles["Normal"],   fontSize=8,
                               textColor=colors.grey)
    normal   = styles["Normal"]
    blue     = colors.HexColor("#2196F3")
    light    = colors.HexColor("#F5F5F5")

    # Header
    story.append(Paragraph(itinerary.get("title", "Trip Itinerary"), title_s))
    story.append(Paragraph(
        f"{prefs.get('source')} → {prefs.get('destination')} | "
        f"{prefs.get('start_date')} to {prefs.get('end_date')} | "
        f"{prefs.get('travelers')} traveler(s)",
        normal,
    ))
    story += [Spacer(1, 8), HRFlowable(width="100%", thickness=1, color=colors.grey), Spacer(1, 8)]

    # Overview
    story.append(Paragraph("Trip Overview", h2_s))
    story.append(Paragraph(itinerary.get("overview", ""), normal))
    story.append(Spacer(1, 8))

    # Budget table
    story.append(Paragraph("Budget Summary", h2_s))
    bdata = [
        ["Item", "Cost (INR)"],
        ["Transport",     f"Rs.{budget.get('transport_cost', 0):,}"],
        ["Accommodation", f"Rs.{budget.get('hotel_cost', 0):,}"],
        ["Daily Variable", f"Rs.{budget.get('daily_variable', 0):,}/day"],
        ["Total Estimate", f"Rs.{budget.get('total_estimate', 0):,}"],
        ["Your Budget",   f"Rs.{budget.get('total_budget', 0):,}"],
    ]
    t = Table(bdata, colWidths=[10*cm, 6*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), blue),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]),
    ]))
    story += [t, Spacer(1, 12)]

    # Day-by-day
    story.append(Paragraph("Day-by-Day Itinerary", h2_s))
    for day in itinerary.get("days", []):
        story.append(Paragraph(
            f"<b>Day {day.get('day')} — {day.get('date')} | {day.get('theme')}</b>", normal
        ))
        for act in day.get("activities", []):
            story.append(Paragraph(
                f"&nbsp;&nbsp;{act.get('time', '')} &nbsp; "
                f"<b>{act.get('location')}</b> — {act.get('activity')}",
                normal,
            ))
        meals = day.get("meals", {})
        if meals:
            story.append(Paragraph(
                f"&nbsp;&nbsp;🍽 B: {meals.get('breakfast','?')} | "
                f"L: {meals.get('lunch','?')} | D: {meals.get('dinner','?')}",
                normal,
            ))
        story.append(Spacer(1, 6))

    # Packing list
    plist = itinerary.get("packing_list", [])
    if plist:
        story.append(Paragraph("Packing List", h2_s))
        cols = 3
        rows = [plist[i:i+cols] for i in range(0, len(plist), cols)]
        if rows and len(rows[-1]) < cols:
            rows[-1] += [""] * (cols - len(rows[-1]))
        t2 = Table(rows)
        t2.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey)]))
        story += [t2, Spacer(1, 8)]

    # Emergency contacts
    ec = itinerary.get("emergency_contacts", {})
    if ec:
        story.append(Paragraph("Emergency Contacts", h2_s))
        for k, v in ec.items():
            story.append(Paragraph(f"&nbsp;&nbsp;{k.replace('_',' ').title()}: {v}", normal))
        story.append(Spacer(1, 8))

    # Footer
    story += [
        HRFlowable(width="100%", thickness=0.5, color=colors.grey),
        Paragraph(
            f"Generated by Trip Planner v3 on {datetime.now().strftime('%d %b %Y %H:%M')}",
            footer_s,
        ),
    ]

    doc.build(story)
    return True
