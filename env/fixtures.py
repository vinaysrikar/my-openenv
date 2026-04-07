"""
fixtures.py — Realistic email datasets for the three task difficulties.
"""
from __future__ import annotations
from env.models import EmailMessage


EASY_INBOX: list[EmailMessage] = [
    EmailMessage(
        id="e1",
        sender="boss@company.com",
        subject="Q3 Report Review — needs your input by Friday",
        body=(
            "Hi,\n\nPlease review the attached Q3 financial report before our Friday standup. "
            "We need your comments on the revenue section. Let me know if you have questions.\n\nThanks"
        ),
        timestamp="2024-11-01T09:00:00",
        true_label="work",
        requires_reply=False,
    ),
    EmailMessage(
        id="e2",
        sender="mom@gmail.com",
        subject="Dinner this Sunday?",
        body=(
            "Hey sweetheart! Are you free for dinner on Sunday? "
            "Dad and I are making your favourite pasta. Let us know!\n\nLove, Mom"
        ),
        timestamp="2024-11-01T08:30:00",
        true_label="personal",
        requires_reply=False,
    ),
    EmailMessage(
        id="e3",
        sender="noreply@discount-pills-now.xyz",
        subject="YOU WON! Claim your $500 Amazon gift card NOW!!!",
        body=(
            "Congratulations!!! You have been selected as today's winner. "
            "Click here to claim your prize: http://totally-not-phishing.xyz/claim "
            "Limited time offer!!! Act NOW!!!"
        ),
        timestamp="2024-11-01T07:45:00",
        true_label="spam",
        is_spam=True,
        requires_reply=False,
    ),
    EmailMessage(
        id="e4",
        sender="weekly@techcrunch.com",
        subject="TechCrunch Weekly Digest — Nov 1",
        body=(
            "This week in tech: OpenAI announces new model, Apple sales rise, "
            "and your weekly roundup of the most important stories in technology. "
            "Unsubscribe | View in browser"
        ),
        timestamp="2024-11-01T06:00:00",
        true_label="newsletter",
        is_newsletter=True,
        requires_reply=False,
    ),
    EmailMessage(
        id="e5",
        sender="sysalert@infra.company.com",
        subject="CRITICAL: Production database down — immediate action required",
        body=(
            "ALERT: The production PostgreSQL cluster has gone down at 08:42 UTC. "
            "Estimated 2,000 users affected. On-call engineer: please acknowledge and begin incident response immediately. "
            "Dashboard: https://status.internal/db-cluster"
        ),
        timestamp="2024-11-01T08:42:00",
        true_label="urgent",
        requires_reply=False,
    ),
]


MEDIUM_INBOX: list[EmailMessage] = EASY_INBOX + [
    EmailMessage(
        id="m1",
        sender="client@acme-corp.com",
        subject="Invoice #4821 — payment question",
        body=(
            "Hello,\n\nWe received Invoice #4821 for $3,200. "
            "Could you confirm that the bank details on the invoice are correct before we process the payment? "
            "We want to make sure funds go to the right account.\n\nBest regards,\nAlex Turner, ACME Corp"
        ),
        timestamp="2024-11-01T10:15:00",
        true_label="work",
        requires_reply=True,
    ),
    EmailMessage(
        id="m2",
        sender="hr@company.com",
        subject="Action required: Benefits enrollment closes Nov 5",
        body=(
            "Hi Team,\n\nThis is a reminder that open enrollment for 2025 benefits closes on November 5th. "
            "Please log into the HR portal and make your selections. "
            "Reply to this email if you need assistance.\n\nHR Team"
        ),
        timestamp="2024-11-01T09:30:00",
        true_label="work",
        requires_reply=True,
    ),
]


HARD_INBOX: list[EmailMessage] = [
    EmailMessage(
        id="h1",
        sender="cto@company.com",
        subject="URGENT: Security breach — all-hands response NOW",
        body=(
            "We have detected unauthorised access to our customer data repository. "
            "This is a P0 incident. All engineers report to #incident-response Slack immediately. "
            "Do NOT discuss externally. Acknowledge receipt of this email."
        ),
        timestamp="2024-11-01T11:00:00",
        true_label="urgent",
        requires_reply=True,
    ),
    EmailMessage(
        id="h2",
        sender="noreply@promo-blasts.net",
        subject="50% OFF everything — today only! Buy Buy Buy!",
        body=(
            "Flash sale! 50% off all products. Use code SPAM50. "
            "Visit http://suspicious-deals.net. Unsubscribe here."
        ),
        timestamp="2024-11-01T07:00:00",
        true_label="spam",
        is_spam=True,
        is_newsletter=False,
        requires_reply=False,
    ),
    EmailMessage(
        id="h3",
        sender="newsletter@medium.com",
        subject="Your weekly Medium digest",
        body=(
            "Top stories this week curated for you. "
            "Story 1: The future of AI. Story 2: Building better habits. "
            "Manage preferences | Unsubscribe"
        ),
        timestamp="2024-11-01T06:30:00",
        true_label="newsletter",
        is_newsletter=True,
        requires_reply=False,
    ),
    EmailMessage(
        id="h4",
        sender="partner@vendor.io",
        subject="Contract renewal — signature needed by Nov 3",
        body=(
            "Hi,\n\nOur annual software contract (ref: VND-2024-881) is due for renewal. "
            "Could you review and sign the attached document by November 3rd? "
            "Please confirm receipt and any questions.\n\nRegards,\nVendor Team"
        ),
        timestamp="2024-11-01T09:45:00",
        true_label="work",
        requires_reply=True,
    ),
    EmailMessage(
        id="h5",
        sender="friend@gmail.com",
        subject="Weekend hiking trip plans",
        body=(
            "Hey! Still on for the hike this Saturday? "
            "Thinking we start at 7am at the trailhead. Let me know if that works!"
        ),
        timestamp="2024-11-01T08:00:00",
        true_label="personal",
        requires_reply=True,
    ),
    EmailMessage(
        id="h6",
        sender="alerts@pagerduty.com",
        subject="[PagerDuty] CRITICAL — API latency exceeding 10s",
        body=(
            "Incident #PD-99321 triggered. Service: payments-api. "
            "Latency p99 = 11.2s (threshold: 2s). Assigned to on-call engineer. "
            "Acknowledge at: https://pagerduty.com/incidents/99321"
        ),
        timestamp="2024-11-01T10:50:00",
        true_label="urgent",
        requires_reply=False,
    ),
    EmailMessage(
        id="h7",
        sender="noreply@linkedin.com",
        subject="You appeared in 12 searches this week",
        body=(
            "Your profile was found in 12 searches this week. "
            "Upgrade to Premium to see who's looking. "
            "Manage email preferences | Unsubscribe"
        ),
        timestamp="2024-11-01T07:30:00",
        true_label="newsletter",
        is_newsletter=True,
        requires_reply=False,
    ),
    EmailMessage(
        id="h8",
        sender="accounts@supplier.com",
        subject="Invoice overdue — second notice",
        body=(
            "Dear Customer,\n\nThis is our second notice regarding Invoice #SUP-2209 for $780, "
            "now 14 days overdue. Please arrange payment or contact us to discuss. "
            "Continued non-payment may result in service suspension.\n\nAccounts Team"
        ),
        timestamp="2024-11-01T09:00:00",
        true_label="work",
        requires_reply=True,
    ),
    EmailMessage(
        id="h9",
        sender="no-reply@spam-lottery.biz",
        subject="Your lottery ticket has WON $1,000,000!!!",
        body=(
            "CONGRATULATIONS! Your email was selected in our international lottery. "
            "To claim $1,000,000 wire $500 processing fee to: spam@scam.biz. "
            "URGENT — claim within 24 hours!"
        ),
        timestamp="2024-11-01T05:00:00",
        true_label="spam",
        is_spam=True,
        requires_reply=False,
    ),
    EmailMessage(
        id="h10",
        sender="team@company.com",
        subject="Team lunch Wednesday — please RSVP",
        body=(
            "Hi everyone,\n\nWe're doing a team lunch this Wednesday at 12:30 at Sakura. "
            "Please reply with your dietary requirements by Tuesday EOD.\n\nCheers,\nThe Team"
        ),
        timestamp="2024-11-01T09:15:00",
        true_label="work",
        requires_reply=True,
    ),
]
