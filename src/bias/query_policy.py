"""Requests whose required evidence cannot be supplied by a public archive.

These guards concern the requested relationship, not the presence of a person's
name. Ordinary published opinions about chatbots, leadership and privacy remain
searchable. They do not substitute for entailment review of other claims.
"""
from __future__ import annotations

import re

_USER_OBJECT = re.compile(r"\b(?:my|our)\b|\bthis\s+(?:chatbot|app|service|product|project)\b|(?:내|우리|저의|제)(?:가\s*(?:만든|개발한|운영하는))?\s*(?:챗봇|앱|서비스|제품|프로젝트)|이\s*(?:챗봇|앱|서비스|제품)", re.I)
_ENDORSEMENT = re.compile(r"\b(?:endors\w*|approv\w*|certif\w*|authoriz\w*|officially\s+(?:support\w*|recommend\w*))\b|승인|보증|인증|공식\s*(?:추천|지지)|추천했", re.I)
_PRIVATE_CONTACT = re.compile(r"\b(?:private|personal|secret)\s+(?:phone|telephone|e-?mail|address|password|messages?)\b|\bhome\s+address\b|(?:개인|비공개)\s*(?:전화|휴대폰|이메일|메일|주소|메시지)|집\s*주소|비밀번호", re.I)
_PRIVATE_USER_EVENT = re.compile(r"\b(?:my|our)\s+private\s+(?:meeting|conversation)\b|(?:나와|저와|우리와|내가)\s*.{0,18}비공개\s*(?:회의|대화)", re.I)


def public_evidence_limitation(query: str) -> str | None:
    if _USER_OBJECT.search(query) and _ENDORSEMENT.search(query):
        return "A public-source archive cannot verify a personal endorsement or approval of the user's product."
    if _PRIVATE_CONTACT.search(query) or _PRIVATE_USER_EVENT.search(query):
        return "This public-source archive cannot verify private contact details or the user's private interactions."
    return None
