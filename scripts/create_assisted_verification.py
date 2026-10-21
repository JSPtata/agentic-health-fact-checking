"""Reconstruct this claim's assisted verification locally."""
import argparse
import hashlib
import json
from pathlib import Path

from verification_schemas import Verification
from verify_claim_agent import dump, validate_verification


EXPECTED_BUNDLE = (
    "320bf14b933570f44ab365ddacbe85bdf"
    "47aa3fbd6bf877a654387f46d3d500c"
)

# Case-specific reviewed data, not a verifier for new claims.
# Each entry contains:
# type, population/protocol, outcome, finding, relation,
# limitation, exact quotation, review resolution.
CARDS = {
    "26275649": (
        "Randomized controlled trial",
        "60 elderly adults with increased pulse pressure; yoga "
        "versus brisk walking plus stretching for 12 weeks, "
        "scheduled in the morning.",
        "Arterial stiffness, endothelial-function indices, "
        "blood pressure and autonomic measures.",
        "The abstract favors yoga over brisk walking plus "
        "stretching. Between-group changes cannot be assigned "
        "to isolated stretching; the walking/stretching arm "
        "had no significant within-group changes.",
        "background",
        "Combined intervention and surrogate vascular outcomes; "
        "morning scheduling does not document timing relative "
        "to waking. Nonsignificance is not proof of no effect.",
        "There was significant change within-yoga group in "
        "vascular function, BP and autonomic function, while "
        "no significant change within-BW group was observed.",
        "Correct the intervention-arm attribution. Yoga "
        "outperforming the combined comparator is not direct "
        "refutation of post-waking stretching.",
    ),
    "38741608": (
        "Animal stretching experiment",
        "Old male Fischer 344 rats; splinted ankle-flexor "
        "stretching, 30 min/day, five days/week for four weeks; "
        "contralateral control limb.",
        "Tibial bone blood flow, remodeling and intravascular volume.",
        "Acute stretching increased local bone blood flow in "
        "naive rats. Training also altered exercise blood flow "
        "and bone structure. This is animal, tissue-specific evidence.",
        "support",
        "Animal findings cannot establish universal human benefit. "
        "Intravascular volume is not interchangeable with flow.",
        "In naive rats, static stretching increased blood flow "
        "to the proximal tibial metaphasis.",
        "Restore rat population and acute/chronic context. "
        "Human/timing gaps limit applicability; they do not make "
        "an accurately qualified animal finding erroneous.",
    ),
    "28397006": (
        "Human acute physiological experiment",
        "25 healthy young men; six 30-second passive calf "
        "stretches; stretched and nonstretched legs compared.",
        "Pulse-wave velocity as arterial stiffness, blood "
        "pressure and heart rate.",
        "Regional PWV decreased in the stretched leg immediately "
        "and at 15 minutes. This indicates a regional stiffness "
        "response, not a demonstrated systemic flow increase.",
        "background",
        "PWV is not blood flow. A contralateral comparator "
        "exists; both legs were assessed.",
        "CONCLUSIONS: These results suggest that mechanical "
        "stimulation to peripheral arteries as induced by static "
        "passive stretch may modulate arterial wall properties "
        "directly, rather than resulting in a systemic effect.",
        "Remove unsupported contradiction labels and restore "
        "full outcome names. The nonstretched leg provides "
        "a comparator.",
    ),
    "41361156": (
        "Controlled trial with randomized groups",
        "54 medicated elderly hypertensive adults; morning/evening "
        "aerobic training versus passive-stretching controls "
        "over 10 weeks.",
        "Blood pressure, systemic hemodynamics and "
        "vascular-function measures.",
        "Aerobic fitness improved in training arms, but the "
        "abstract reports no significant hemodynamic/vascular "
        "changes or adjusted group differences. It does not "
        "establish a waking-specific stretching effect.",
        "null_or_uncertain",
        "Passive stretching was a control intervention. "
        "Nonsignificance does not demonstrate equivalence "
        "or absence of effect.",
        "The effects of the interventions adjusted for the "
        "pre-intervention values also showed no differences "
        "among the groups (all, P\u2009>\u20090.05).",
        "Restore the passive-stretching control group. Retain "
        "null-result uncertainty and the timing gap without "
        "converting either into refutation.",
    ),
    "28780647": (
        "Review; systematic methods not established by supplied abstract",
        "Literature on skeletal-muscle stretching and cardiovascular "
        "responses; not a new participant cohort.",
        "Acute circulatory responses and possible longer-term "
        "vascular effects.",
        "The review describes vascular responses and proposes "
        "possible therapeutic benefits. It supplies mechanistic "
        "context rather than direct evidence for the "
        "post-waking recommendation.",
        "background",
        "Review overlap with primary studies was not checked; "
        "do not count it as another independent experiment "
        "or as a clinical guideline.",
        "The current evidence suggests that acute stretching "
        "causes a significant macro- and microcirculatory event "
        "that alters blood flow and the relationship between "
        "oxygen availability and oxygen utilization.",
        "The described vascular responses are not contradiction. "
        "Preserve review status and distinguish potential "
        "benefits from demonstrated clinical outcomes.",
    ),
    "25324203": (
        "Small human physiological study",
        "Nine recreationally fit adults; four-minute self-administered "
        "plantar-flexor stretching and cycling protocols.",
        "NIRS deoxygenated hemoglobin/myoglobin and muscle "
        "blood-volume measures.",
        "The authors infer local poststretch hyperemia from "
        "NIRS blood-volume changes at 1, 5 and 10 minutes. "
        "This supports a limited physiological possibility, "
        "not an established whole-body or post-waking benefit.",
        "support",
        "Flow was inferred from NIRS blood-volume changes, "
        "not directly measured volumetric flow. The abstract "
        "also mentions a 60-second event; do not infer an "
        "optimal dose.",
        "Furthermore, the increase in local blood flow "
        "(suggested by [Hbtot ] changes) was found to be "
        "significantly increased relative to baseline at "
        "1, 5 and 10 min after SAS, thus providing novel "
        "evidence for a poststretch hyperaemia.",
        "Correct the contradiction label and preserve the "
        "inferred, local nature of the flow interpretation.",
    ),
    "33337692": (
        "Acute human warm-up comparison",
        "40 healthy adults aged 18-25; running, plyometrics, "
        "eccentric heel drops and static stretching.",
        "Achilles tendon blood flow and stiffness.",
        "Running and plyometrics increased measured tendon "
        "outcomes; static stretching and eccentric exercise "
        "showed no significant changes.",
        "null_or_uncertain",
        "Tendon-specific acute results do not establish a "
        "universal effect. Nonsignificant findings are not "
        "equivalence evidence.",
        "Static stretching and eccentric exercises elicited "
        "no significant changes.",
        "Static stretching was explicitly tested. Remove "
        "claims of no stretching comparison; tendon is an "
        "anatomical target, not the participant population.",
    ),
    "32596231": (
        "Comparative human microvascular study",
        "30 men with type 2 diabetes, 15 smokers and 15 nonsmokers, "
        "plus 15 normal nonsmoking controls; passive ankle dorsiflexion.",
        "Calf skin blood flow measured using laser Doppler flowmetry.",
        "All subjects showed poststretch reactive hyperemia "
        "after a 10-second stimulus, with lower intensity in "
        "diabetic smokers. Responses were local and "
        "population-dependent.",
        "support",
        "A local skin microvascular response does not establish "
        "systemic or clinical benefit. Non-diabetic controls "
        "and acute responses were present.",
        "CONCLUSION: All subjects expressed the PSRH effect "
        "in cutaneous microcirculation after a 10-s stretch "
        "stimulus; however, this effect was observed at a "
        "significantly lower intensity in chronic smokers "
        "with diabetes.",
        "Correct claims that controls or acute effects were "
        "absent. Preserve all three groups and the "
        "population-dependent response.",
    ),
    "35981730": (
        "Human four-session physiological experiment",
        "Eight healthy young men; passive leg movement with "
        "contralateral resting/stretching conditions, with or "
        "without exercise-induced muscle damage.",
        "Femoral blood flow, leg vascular conductance and "
        "central hemodynamics.",
        "Muscle damage plus stretching reduced resting limb "
        "flow and movement-induced hyperemia while increasing "
        "central hemodynamics. This is a context-specific "
        "counterpattern, not evidence of a general post-waking harm.",
        "contradiction",
        "Direction applies only to the experimental condition. "
        "Increased central pressure/output is not equivalent "
        "to increased peripheral flow.",
        "The combination of static stretching protocol on "
        "the damaged limb extensively increases resting "
        "central hemodynamics with reduction in resting limb "
        "blood flow and passive leg movement-induced hyperemia.",
        "Participants were eight men, not rats. Reduced limb "
        "flow cannot be labeled support for increased flow. "
        "Retain the restricted muscle-damage context.",
    ),
    "31563842": (
        "Human muscle-length physiology experiment",
        "10 healthy young adults; stretched versus relaxed "
        "vastus lateralis and gastrocnemius positions.",
        "Endurance, NIRS oxygen saturation, blood-flow measures "
        "and oxygen-recovery timing.",
        "Endurance was lower and oxygen recovery slower in "
        "stretched positions; oxygen saturation during endurance "
        "tests was not significantly different. The abstract "
        "does not separately establish a directional flow increase.",
        "background",
        "Oxygenation/recovery are not direct flow outcomes. "
        "Blood flow was listed as measured; absence of a clear "
        "reported result is not absence of measurement.",
        "However, oxygen saturation during the endurance tests "
        "were not different between stretched and relaxed "
        "conditions in both muscle (p\u202f>\u202f0.05 for "
        "all comparisons).",
        "Restore full-source sample and methods. Preserve "
        "endpoint distinctions and avoid treating missing "
        "timing as an error in an accurately limited finding.",
    ),
    "33079234": (
        "Human acute experiment with contralateral-leg comparison",
        "Nine healthy young men; six 30-second passive calf "
        "stretches with 10-second relaxation intervals; "
        "both legs measured.",
        "Posterior tibial artery diameter, velocity, flow "
        "and shear rate measured by Doppler ultrasound.",
        "Mean velocity and shear rate increased in relaxation "
        "intervals between stretches. They did not change "
        "during the stretch phase; the nonstretched leg "
        "was unchanged.",
        "support",
        "Velocity/shear are not volumetric flow. Between-repetition "
        "relaxation is not the same as after the complete session; "
        "a separate flow result was not quantified in the abstract.",
        "CONCLUSIONS: The stretching procedure increased the "
        "shear rate in the peripheral artery of the stretched "
        "leg during the relaxation period.",
        "Replace the finding containing only Q1 with a meaningful "
        "result. Correct unsupported contradiction labels, phase "
        "interpretation and claims that only one leg was measured.",
    ),
    "41467762": (
        "Rat stretching/sham experiment",
        "20 male Wistar rats; varied soleus stretching intensity "
        "and a sustained two-hour stretch condition.",
        "Microvascular oxygen partial pressure and HIF-1alpha expression.",
        "Stretching decreased microvascular oxygen tension; "
        "sustained stretching also reduced HIF-1alpha expression. "
        "This is not evidence for increased human blood flow.",
        "background",
        "Animal oxygen-tension measurements are not human flow "
        "outcomes. Formula placeholders in the extracted "
        "abstract limit interpretation.",
        "Microvascular Po<sub>2</sub> decreased in an "
        "intensity-dependent manner and remained low "
        "during prolonged stretching.",
        "Restore rat population and correct endpoint/direction "
        "interpretation. Reduced oxygen tension cannot be "
        "labeled support for increased circulation.",
    ),
    "who.int": (
        "WHO physical-activity fact sheet",
        "General population guidance on activity, inactivity "
        "and public-health policy.",
        "General activity recommendations and health associations.",
        "This extracted page provides general activity guidance, "
        "without a specific recommendation or experiment "
        "establishing post-waking stretching for circulation.",
        "background",
        "Absence on this page is not absence across WHO guidance. "
        "Listed references were not separately assessed "
        "as primary studies.",
        "WHO defines physical activity as any bodily movement "
        "produced by skeletal muscles that requires energy expenditure.",
        "General guidance can be accurately classified as "
        "background. The reference list is not a new pooled "
        "analysis; adult 507-survey and adolescent 298-survey "
        "citations must not be conflated.",
    ),
    "nhs.uk": (
        "NHS adult physical-activity guidance",
        "Adults aged 19-64; general aerobic and muscle-strengthening advice.",
        "Activity recommendations, not measured "
        "stretching-related circulation outcomes.",
        "The extracted page recommends regular activity and "
        "individual appropriateness. It does not establish a "
        "post-waking stretching recommendation for circulation.",
        "background",
        "This finding concerns only the supplied page. Linked "
        "guidance and its applicability were not separately assessed.",
        "Adults should do some type of physical activity every "
        "day. Exercise just once or twice a week can reduce "
        "the risk of heart disease or stroke.",
        "Circulation is not a measured outcome of this guidance "
        "page. Retain background status; lack of specific "
        "evidence does not make a correctly qualified "
        "description erroneous.",
    ),
}


def build(folder):
    bundle_path = folder / "verification-review-bundle.json"
    raw = bundle_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    if digest != EXPECTED_BUNDLE:
        raise ValueError(
            "This transfer applies only to the exact review "
            "bundle uploaded in this conversation"
        )

    bundle = json.loads(raw)
    original = bundle["report"]
    registry = original["source_registry"]
    texts = bundle["source_texts"]

    by_key = {
        str(
            registry[sid].get("pmid")
            or registry[sid]["organization"]
        ): sid
        for sid in original["active_sources"]
    }

    if (
        set(by_key) != set(CARDS)
        or set(texts) != set(by_key.values())
    ):
        raise ValueError("Unexpected source coverage")

    hashes = {
        sid: hashlib.sha256(text.encode()).hexdigest()
        for sid, text in texts.items()
    }

    if (
        hashes != original["binding"]["source_hashes"]
        or any(
            hashes[sid] != registry[sid]["text_sha256"]
            for sid in texts
        )
    ):
        raise ValueError(
            "Source hashes do not match the reviewed evidence"
        )

    original_bytes = json.dumps(
        original, indent=2, ensure_ascii=False
    ).encode()

    if (
        hashlib.sha256(original_bytes).hexdigest()
        != bundle["report_sha256"]
    ):
        raise ValueError("Original report digest mismatch")

    findings = []

    for key, card in CARDS.items():
        (
            kind,
            population,
            outcome,
            finding,
            relation,
            limitation,
            quotation,
            _,
        ) = card

        sid = by_key[key]

        findings.append({
            "source_id": sid,
            "study_or_document_type": kind,
            "population_and_intervention": population,
            "measured_outcome_or_recommendation": outcome,
            "finding": finding,
            "relation": relation,
            "limitations": [
                limitation,
                "The supplied source does not establish the "
                "claimed post-waking routine or universal "
                "always recommendation.",
            ],
            "quotes": [quotation],
        })

    verification = Verification.model_validate({
        "claim_id": original["claim"]["claim_id"],
        "findings": findings,
        "components": [
            {
                "component": "exposure_outcome",
                "conclusion": "mixed",
                "source_ids": [
                    by_key[key]
                    for key in (
                        "25324203",
                        "33337692",
                        "32596231",
                        "35981730",
                        "33079234",
                    )
                ],
                "explanation": (
                    "Some protocols show local vascular responses; "
                    "others show nonsignificant changes or "
                    "context-specific reduced flow. Endpoints, "
                    "populations and phases differ. This is "
                    "qualitative heterogeneity, not a pooled "
                    "effect estimate."
                ),
            },
            {
                "component": "context",
                "conclusion": "insufficient_evidence",
                "source_ids": list(texts),
                "explanation": (
                    "The supplied sources do not establish an "
                    "isolated post-waking stretching routine. "
                    "Morning scheduling alone does not document "
                    "timing relative to waking. No exact delay or "
                    "superiority over other times is imposed."
                ),
            },
            {
                "component": "recommendation_strength",
                "conclusion": "insufficient_evidence",
                "source_ids": list(texts),
                "explanation": (
                    "Selected physiological experiments and "
                    "general activity guidance do not justify "
                    "always. This does not establish that "
                    "stretching is unsafe or ineffective."
                ),
            },
        ],
        "provisional_verdict": "insufficient_evidence",
        "summary": (
            "Stretching can produce local vascular responses "
            "under particular conditions, but the supplied "
            "evidence does not establish that everyone should "
            "always stretch after waking to boost circulation. "
            "Missing direct support is not proof that the "
            "claim is false."
        ),
        "unresolved_gaps": [
            "A directly evaluated post-waking stretching routine "
            "with an appropriate comparator is absent from this "
            "supplied active evidence set.",
            "Optimal dose, response duration, clinical relevance "
            "and general applicability are not established.",
            "Only 12 abstracts and two extracted official pages "
            "were reviewed; 21 inactive registry entries were "
            "not assessed here.",
            "Search completeness, full-text risk of bias, study "
            "overlap and retraction/correction status were not "
            "independently audited.",
            "This is an assistant-assisted correction, not "
            "independent expert validation or a successful "
            "autonomous rerun.",
        ],
    })

    validate_verification(
        verification,
        original["claim"]["claim_id"],
        {
            sid: {"text": text}
            for sid, text in texts.items()
        },
    )

    key_for_id = {
        sid: key for key, sid in by_key.items()
    }

    resolutions = [
        {
            "source_id": item["source_id"],
            "part": item["part"],
            "was_blocked": (
                item["review"]["decision"] == "revise"
                or any(
                    objection["severity"] in ("major", "critical")
                    for objection in item["review"]["objections"]
                )
            ),
            "category": "assistant_whole_source_correction",
            "resolution": CARDS[
                key_for_id[item["source_id"]]
            ][7],
            "disposition": (
                "Use the corrected whole-source finding in "
                "this assisted report; original passage "
                "assessment is preserved."
            ),
        }
        for item in original["passage_assessments"]
    ]

    result = {
        "status": "assisted_verification_ready_for_adjudication",
        "claim": original["claim"],
        "verification": verification.model_dump(),
        "provenance": {
            "method": (
                "Compact local reconstruction of this "
                "conversation's assistant-reviewed corrections; "
                "no new model run."
            ),
            "bundle_sha256": digest,
            "source_hashes": hashes,
            "original_report_sha256": bundle["report_sha256"],
            "independent_expert_review": False,
            "source_data_independently_refetched": False,
            "original_local_model": original["binding"]["model"],
            "transfer_version": "compact-assisted-report-v1",
        },
        "review_reconciliation": resolutions,
        "source_registry": registry,
        "source_texts": texts,
        "original_report": original,
        "limitations": (
            original["limitations"]
            + verification.unresolved_gaps
        ),
    }

    output = folder / "verification-assisted.json"

    if output.exists() and output.read_bytes().strip():
        if json.loads(output.read_text()) != result:
            raise FileExistsError(
                "A different assisted report already exists; "
                "preserve it before creating another version"
            )
    else:
        dump(output, result)

    print("CREATED:", output.resolve())
    print("VERDICT:", verification.provisional_verdict)
    print("SOURCES:", len(findings))
    print("RECONCILED PASSAGES:", len(resolutions))
    print("No API calls made. Original report preserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--run-dir", required=True, type=Path
    )
    build(parser.parse_args().run_dir)