from typing import TypedDict, Optional


class TripState(TypedDict, total=False):
    # ── Session ───────────────────────────────────────────────────────────────
    session_id:             str       # uuid4 — LangGraph thread_id for SqliteSaver
    checkpoint_path:        str       # checkpoints/trip_{session_id}.db
    log_path:               str       # logs/session_{session_id}.log
    last_completed_node:    str       # last node that finished successfully
    cache_hits:             dict      # {geocode: bool, weather: bool, places: bool}

    # ── Core preferences (filled by Agent 1) ─────────────────────────────────
    user_profile:           dict      # {user_id, name}
    trip_preferences:       dict      # {source, destination, start_date, end_date,
                                      #  budget, travelers, travel_type, hotel_pref,
                                      #  food_pref, transport_pref}

    # ── Memory (filled by Agent 2) ────────────────────────────────────────────
    memory_context:         dict      # {past_trips[], preferred_hotel, avg_budget,
                                      #  favourite_destinations[], notes}

    # ── Geocoding (filled once before parallel agents) ────────────────────────
    coordinates:            dict      # {source_lat, source_lon, dest_lat, dest_lon}

    # ── Weather advisory (filled by Weather Agent + Gate 0 HITL) ─────────────
    weather_data:           dict      # {condition, summary, concerns, score,
                                      #  raw_forecast[]}
    weather_advisory:       dict      # {triggered, alternatives[], user_choice,
                                      #  destination_changed: bool}

    # ── Live API outputs ──────────────────────────────────────────────────────
    transport_data:         dict      # {final_mode, flight_offer{}, train_option{},
                                      #  estimated_cost_per_person, duration_hours}
    hotel_data:             dict      # {recommended{name, id, price_per_night,
                                      #  total}, top_3[], amadeus_hotel_id}
    places_data:            dict      # {must_visit[], food_spots[], activities[],
                                      #  hidden_gems[], source: foursquare|llm}
    places_confirmed:       dict      # set after Gate 2 — user-approved place list
                                      # {must_visit[], food_spots[], activities[],
                                      #  hidden_gems[], user_added[], user_removed[]}

    # ── Budget ────────────────────────────────────────────────────────────────
    budget_summary:         dict      # {breakdown{}, total_estimate, within_budget,
                                      #  daily_pool, base_daily, savings_tips[]}
    daily_budget_targets:   list      # [{day, date, day_type, weight,
                                      #  target_inr, min_inr, max_inr}]
    budget_revisions:       list      # Gate 3 history: [{round, action, old_total,
                                      #  new_total, swaps[]}]
    budget_gate_round:      int       # current Gate 3 round (max 3)

    # ── Itinerary + Review ────────────────────────────────────────────────────
    itinerary:              dict      # {title, overview, days[], packing_list,
                                      #  emergency_contacts}
    review_status:          dict      # {passed, score, issues[], warnings[],
                                      #  positive_highlights[], overall_verdict}

    # ── Evaluations + Guardrails ──────────────────────────────────────────────
    eval_scores:            dict      # {pacing, variety, preference_alignment,
                                      #  distance_vs_budget, overall}
    guardrail_results:      dict      # {input_valid, budget_conflict, output_valid,
                                      #  warnings[]}

    # ── API status tracking ───────────────────────────────────────────────────
    api_status:             dict      # {weather, flights, trains, hotels, places}
                                      # each = "live" | "fallback"

    # ── Orchestrator + HITL ───────────────────────────────────────────────────
    orchestrator_decision:  dict      # {approved, retry_agents[], reason,
                                      #  force_stop}
    hitl_approved:          bool      # True when user approved at current gate
    hitl_gate:              str       # gate0|gate1|gate2|gate3|gate4|gate5
    human_feedback:         str       # free-text change request from user
    hitl_change_count:      int       # total times user requested changes

    # ── Retry + Errors ────────────────────────────────────────────────────────
    retry_count:            int       # current orchestrator retry (max 3)
    failed_agents:          list      # agents that exhausted all 3 retries

    # ── Output ────────────────────────────────────────────────────────────────
    pdf_path:               Optional[str]
    whatsapp_summary_path:  Optional[str]
    error:                  Optional[str]
