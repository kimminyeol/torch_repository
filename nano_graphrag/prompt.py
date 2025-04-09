# """
# Reference:
#  - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
# """

# GRAPH_FIELD_SEP = "<SEP>"
# PROMPTS = {}

# PROMPTS[
#     "claim_extraction"
# ] = """-Target activity-
# You are an intelligent assistant that helps a human analyst to analyze claims against certain entities presented in a text document.

# -Goal-
# Given a text document that is potentially relevant to this activity, an entity specification, and a claim description, extract all entities that match the entity specification and all claims against those entities.

# -Steps-
# 1. Extract all named entities that match the predefined entity specification. Entity specification can either be a list of entity names or a list of entity types.
# 2. For each entity identified in step 1, extract all claims associated with the entity. Claims need to match the specified claim description, and the entity should be the subject of the claim.
# For each claim, extract the following information:
# - Subject: name of the entity that is subject of the claim, capitalized. The subject entity is one that committed the action described in the claim. Subject needs to be one of the named entities identified in step 1.
# - Object: name of the entity that is object of the claim, capitalized. The object entity is one that either reports/handles or is affected by the action described in the claim. If object entity is unknown, use **NONE**.
# - Claim Type: overall category of the claim, capitalized. Name it in a way that can be repeated across multiple text inputs, so that similar claims share the same claim type
# - Claim Status: **TRUE**, **FALSE**, or **SUSPECTED**. TRUE means the claim is confirmed, FALSE means the claim is found to be False, SUSPECTED means the claim is not verified.
# - Claim Description: Detailed description explaining the reasoning behind the claim, together with all the related evidence and references.
# - Claim Date: Period (start_date, end_date) when the claim was made. Both start_date and end_date should be in ISO-8601 format. If the claim was made on a single date rather than a date range, set the same date for both start_date and end_date. If date is unknown, return **NONE**.
# - Claim Source Text: List of **all** quotes from the original text that are relevant to the claim.

# Format each claim as (<subject_entity>{tuple_delimiter}<object_entity>{tuple_delimiter}<claim_type>{tuple_delimiter}<claim_status>{tuple_delimiter}<claim_start_date>{tuple_delimiter}<claim_end_date>{tuple_delimiter}<claim_description>{tuple_delimiter}<claim_source>)

# 3. Return output in English as a single list of all the claims identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

# 4. When finished, output {completion_delimiter}

# -Examples-
# Example 1:
# Entity specification: organization
# Claim description: red flags associated with an entity
# Text: According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B. The company is owned by Person C who was suspected of engaging in corruption activities in 2015.
# Output:

# (COMPANY A{tuple_delimiter}GOVERNMENT AGENCY B{tuple_delimiter}ANTI-COMPETITIVE PRACTICES{tuple_delimiter}TRUE{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10{tuple_delimiter}According to an article published on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B.)
# {completion_delimiter}

# Example 2:
# Entity specification: Company A, Person C
# Claim description: red flags associated with an entity
# Text: According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B. The company is owned by Person C who was suspected of engaging in corruption activities in 2015.
# Output:

# (COMPANY A{tuple_delimiter}GOVERNMENT AGENCY B{tuple_delimiter}ANTI-COMPETITIVE PRACTICES{tuple_delimiter}TRUE{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10{tuple_delimiter}According to an article published on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B.)
# {record_delimiter}
# (PERSON C{tuple_delimiter}NONE{tuple_delimiter}CORRUPTION{tuple_delimiter}SUSPECTED{tuple_delimiter}2015-01-01T00:00:00{tuple_delimiter}2015-12-30T00:00:00{tuple_delimiter}Person C was suspected of engaging in corruption activities in 2015{tuple_delimiter}The company is owned by Person C who was suspected of engaging in corruption activities in 2015)
# {completion_delimiter}

# -Real Data-
# Use the following input for your answer.
# Entity specification: {entity_specs}
# Claim description: {claim_description}
# Text: {input_text}
# Output: """

# PROMPTS[
#     "community_report"
# ] = """You are an AI assistant that helps a human analyst to perform general information discovery. 
# Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

# # Goal
# Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# # Report Structure

# The report should include the following sections:

# - TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
# - SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
# - IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
# - RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
# - DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

# Return output as a well-formed JSON-formatted string with the following format:
#     {{
#         "title": <report_title>,
#         "summary": <executive_summary>,
#         "rating": <impact_severity_rating>,
#         "rating_explanation": <rating_explanation>,
#         "findings": [
#             {{
#                 "summary":<insight_1_summary>,
#                 "explanation": <insight_1_explanation>
#             }},
#             {{
#                 "summary":<insight_2_summary>,
#                 "explanation": <insight_2_explanation>
#             }}
#             ...
#         ]
#     }}

# # Grounding Rules
# Do not include information where the supporting evidence for it is not provided.


# # Example Input
# -----------
# Text:
# ```
# Entities:
# ```csv
# id,entity,type,description
# 5,VERDANT OASIS PLAZA,geo,Verdant Oasis Plaza is the location of the Unity March
# 6,HARMONY ASSEMBLY,organization,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza
# ```
# Relationships:
# ```csv
# id,source,target,description
# 37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
# 38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
# 39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
# 40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
# 41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
# 43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March
# ```
# ```
# Output:
# {{
#     "title": "Verdant Oasis Plaza and Unity March",
#     "summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
#     "rating": 5.0,
#     "rating_explanation": "The impact severity rating is moderate due to the potential for unrest or conflict during the Unity March.",
#     "findings": [
#         {{
#             "summary": "Verdant Oasis Plaza as the central location",
#             "explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes."
#         }},
#         {{
#             "summary": "Harmony Assembly's role in the community",
#             "explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community."
#         }},
#         {{
#             "summary": "Unity March as a significant event",
#             "explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community."
#         }},
#         {{
#             "summary": "Role of Tribune Spotlight",
#             "explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved."
#         }}
#     ]
# }}


# # Real Data

# Use the following text for your answer. Do not make anything up in your answer.

# Text:
# ```
# {input_text}
# ```

# The report should include the following sections:

# - TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
# - SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
# - IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
# - RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
# - DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

# Return output as a well-formed JSON-formatted string with the following format:
#     {{
#         "title": <report_title>,
#         "summary": <executive_summary>,
#         "rating": <impact_severity_rating>,
#         "rating_explanation": <rating_explanation>,
#         "findings": [
#             {{
#                 "summary":<insight_1_summary>,
#                 "explanation": <insight_1_explanation>
#             }},
#             {{
#                 "summary":<insight_2_summary>,
#                 "explanation": <insight_2_explanation>
#             }}
#             ...
#         ]
#     }}

# # Grounding Rules
# Do not include information where the supporting evidence for it is not provided.

# Output:
# """

# PROMPTS[
#     "entity_extraction"
# ] = """-Goal-
# Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

# -Steps-
# 1. Identify all entities. For each identified entity, extract the following information:
# - entity_name: Name of the entity, capitalized
# - entity_type: One of the following types: [{entity_types}]
# - entity_description: Comprehensive description of the entity's attributes and activities
# Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

# 2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
# For each pair of related entities, extract the following information:
# - source_entity: name of the source entity, as identified in step 1
# - target_entity: name of the target entity, as identified in step 1
# - relationship_description: explanation as to why you think the source entity and the target entity are related to each other
# - relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
#  Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

# 3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

# 4. When finished, output {completion_delimiter}

# ######################
# -Examples-
# ######################
# Example 1:

# Entity_types: [person, technology, mission, organization, location]
# Text:
# while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

# Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

# The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

# It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
# ################
# Output:
# ("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
# ("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
# ("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
# ("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
# ("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
# ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
# ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
# ("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
# ("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
# ("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
# #############################
# Example 2:

# Entity_types: [person, technology, mission, organization, location]
# Text:
# They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

# Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

# Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
# #############
# Output:
# ("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
# ("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
# ("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
# ("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}7){record_delimiter}
# ("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}9){completion_delimiter}
# #############################
# Example 3:

# Entity_types: [person, role, technology, organization, event, location, concept]
# Text:
# their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

# "It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

# Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

# Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

# The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
# #############
# Output:
# ("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
# ("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
# ("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
# ("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
# ("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
# ("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
# ("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}9){record_delimiter}
# ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}10){record_delimiter}
# ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}8){record_delimiter}
# ("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
# #############################
# -Real Data-
# ######################
# Entity_types: {entity_types}
# Text: {input_text}
# ######################
# Output:
# """






# PROMPTS[
#     "summarize_entity_descriptions"
# ] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
# Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
# Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
# If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
# Make sure it is written in third person, and include the entity names so we the have full context.

# #######
# -Data-
# Entities: {entity_name}
# Description List: {description_list}
# #######
# Output:
# """


# PROMPTS[
#     "entiti_continue_extraction"
# ] = """MANY entities were missed in the last extraction.  Add them below using the same format:
# """

# PROMPTS[
#     "entiti_if_loop_extraction"
# ] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
# """

# PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]
# PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
# PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
# PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# PROMPTS[
#     "local_rag_response"
# ] = """---Role---

# You are a helpful assistant responding to questions about data in the tables provided.


# ---Goal---

# Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
# If you don't know the answer, just say so. Do not make anything up.
# Do not include information where the supporting evidence for it is not provided.

# ---Target response length and format---

# {response_type}


# ---Data tables---

# {context_data}


# ---Goal---

# Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

# If you don't know the answer, just say so. Do not make anything up.

# Do not include information where the supporting evidence for it is not provided.


# ---Target response length and format---

# {response_type}

# Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
# """

# PROMPTS[
#     "global_map_rag_points"
# ] = """---Role---

# You are a helpful assistant responding to questions about data in the tables provided.


# ---Goal---

# Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

# You should use the data provided in the data tables below as the primary context for generating the response.
# If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

# Each key point in the response should have the following element:
# - Description: A comprehensive description of the point.
# - Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

# The response should be JSON formatted as follows:
# {{
#     "points": [
#         {{"description": "Description of point 1...", "score": score_value}},
#         {{"description": "Description of point 2...", "score": score_value}}
#     ]
# }}

# The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
# Do not include information where the supporting evidence for it is not provided.


# ---Data tables---

# {context_data}

# ---Goal---

# Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

# You should use the data provided in the data tables below as the primary context for generating the response.
# If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

# Each key point in the response should have the following element:
# - Description: A comprehensive description of the point.
# - Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

# The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
# Do not include information where the supporting evidence for it is not provided.

# The response should be JSON formatted as follows:
# {{
#     "points": [
#         {{"description": "Description of point 1", "score": score_value}},
#         {{"description": "Description of point 2", "score": score_value}}
#     ]
# }}
# """

# PROMPTS[
#     "global_reduce_rag_response"
# ] = """---Role---

# You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


# ---Goal---

# Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

# Note that the analysts' reports provided below are ranked in the **descending order of importance**.

# If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

# The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

# Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

# The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

# Do not include information where the supporting evidence for it is not provided.


# ---Target response length and format---

# {response_type}


# ---Analyst Reports---

# {report_data}


# ---Goal---

# Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

# Note that the analysts' reports provided below are ranked in the **descending order of importance**.

# If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

# The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

# The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

# Do not include information where the supporting evidence for it is not provided.


# ---Target response length and format---

# {response_type}

# Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
# """

# PROMPTS[
#     "naive_rag_response"
# ] = """You're a helpful assistant
# Below are the knowledge you know:
# {content_data}
# ---
# If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
# Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
# If you don't know the answer, just say so. Do not make anything up.
# Do not include information where the supporting evidence for it is not provided.
# ---Target response length and format---
# {response_type}
# """

# PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

# PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# PROMPTS["default_text_separator"] = [
#     # Paragraph separators
#     "\n\n",
#     "\r\n\r\n",
#     # Line breaks
#     "\n",
#     "\r\n",
#     # Sentence ending punctuation
#     "。",  # Chinese period
#     "．",  # Full-width dot
#     ".",  # English period
#     "！",  # Chinese exclamation mark
#     "!",  # English exclamation mark
#     "？",  # Chinese question mark
#     "?",  # English question mark
#     # Whitespace characters
#     " ",  # Space
#     "\t",  # Tab
#     "\u3000",  # Full-width space
#     # Special characters
#     "\u200b",  # Zero-width space (used in some Asian languages)
# ]














"""
Reference:
 - Prompts are from [graphrag](https://github.com/microsoft/graphrag)
"""

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS["claim_extraction"] = """-목표 활동-
당신은 문서에 포함된 특정 엔티티(개체)에 대한 주장들을 분석하는 데 도움을 주는 지능형 분석 도우미입니다.

-목표-
주어진 문서, 엔티티 사양(Entity Specification), 주장 설명(Claim Description)을 기반으로 다음 작업을 수행하세요:
1) 엔티티 사양에 부합하는 모든 엔티티를 문서에서 식별하고,
2) 각 엔티티에 대해 해당 주장 설명과 일치하는 모든 주장(Claim)을 추출하세요.

-작업 순서-
1. 엔티티 사양과 일치하는 모든 엔티티를 추출하세요.
   - 엔티티 사양은 엔티티 이름 리스트 또는 엔티티 유형 리스트일 수 있습니다.
   - 예: "삼성전자", "조국", 또는 "조직", "사람", "장소"

2. 추출된 각 엔티티에 대해 다음 항목이 포함된 주장을 추출하세요:
- Subject (주어): 주장의 대상이 되는 엔티티의 이름 (대문자 표기)
- Object (목적어): 주장의 영향을 받거나 처리하는 대상. 해당 엔티티가 없으면 **NONE**
- Claim Type (주장 유형): 동일한 주제의 주장들을 묶을 수 있는 일반화된 카테고리 이름 (대문자 표기)
- Claim Status (주장 상태): **TRUE**(사실로 확인됨), **FALSE**(거짓으로 확인됨), **SUSPECTED**(확인되지 않음) 중 하나
- Claim Description (주장 설명): 주장을 뒷받침하는 상세한 이유와 근거 설명
- Claim Date (주장 시점): ISO-8601 형식의 날짜 혹은 기간 (예: 2024-03-01T00:00:00)
- Claim Source Text (출처 문장): 원문에서 주장을 뒷받침하는 모든 인용문 목록

각 주장은 다음과 같은 형식으로 작성됩니다:
(<subject_entity>{tuple_delimiter}<object_entity>{tuple_delimiter}<claim_type>{tuple_delimiter}<claim_status>{tuple_delimiter}<claim_start_date>{tuple_delimiter}<claim_end_date>{tuple_delimiter}<claim_description>{tuple_delimiter}<claim_source>)

3. 위와 같은 형식으로 추출된 모든 주장들을 리스트 형태로 출력하세요. 리스트의 각 항목은 **{record_delimiter}**로 구분하세요.

4. 모든 출력을 마치면 마지막에 {completion_delimiter}를 출력하세요.

-예시-
예시 1:
엔티티 사양: organization
주장 설명: 조직의 문제점이나 부정적 이슈
문서:
2022년 1월 10일자 기사에 따르면, 회사 A는 정부 기관 B가 발주한 여러 공공입찰에서 담합을 하다가 벌금을 부과받았다. 이 회사는 2015년에 부패 혐의를 받은 인물 C가 소유하고 있다.

출력:
(COMPANY A{tuple_delimiter}GOVERNMENT AGENCY B{tuple_delimiter}ANTI-COMPETITIVE PRACTICES{tuple_delimiter}TRUE{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}Company A는 여러 공공입찰에서 담합하다 벌금을 받아 반경쟁적 행위로 간주됨{tuple_delimiter}2022년 1월 10일자 기사에 따르면, 회사 A는 정부 기관 B가 발주한 여러 공공입찰에서 담합을 하다가 벌금을 부과받았다.)
{completion_delimiter}

예시 2:
엔티티 사양: Company A, Person C
주장 설명: 조직 또는 개인의 부정 행위 또는 의혹
문서:
...

출력:
(...생략...)
{completion_delimiter}

-실제 데이터-
다음 입력을 사용하세요:
Entity specification: {entity_specs}
Claim description: {claim_description}
Text: {input_text}
Output: """




PROMPTS["community_report"] = """당신은 커뮤니티 내 다양한 엔터티(예: 기관, 개인)와 이들 간의 관계 및 관련 주장을 바탕으로, 커뮤니티에 대한 정보를 종합적으로 분석해주는 AI 분석 도우미입니다.

# 목표
주어진 커뮤니티에 속한 엔터티 리스트와 이들 간의 관계 및 (선택적으로) 관련 주장(claims)을 바탕으로, 해당 커뮤니티에 대한 종합적인 보고서를 작성하세요. 이 보고서는 의사결정자에게 커뮤니티 구성 요소의 특성과 영향력을 이해시키는 데 활용됩니다.

보고서에는 다음과 같은 정보가 포함되어야 합니다:
- 주요 엔터티 개요
- 법적/사회적 문제 여부
- 기술적 능력
- 평판
- 주목할 만한 주장 등

# 보고서 구성

보고서는 아래 항목들을 반드시 포함해야 합니다:

- TITLE (제목): 커뮤니티의 대표적인 엔터티 이름을 포함한 간결하고 구체적인 제목
- SUMMARY (요약): 커뮤니티의 전체 구조, 주요 엔터티 간 관계, 핵심 정보에 대한 요약 설명
- IMPACT SEVERITY RATING (영향력 점수): 커뮤니티가 가질 수 있는 영향력 수준 (0~10 사이 소수점 가능)
- RATING EXPLANATION (점수 설명): 위 점수에 대한 한 줄 설명
- DETAILED FINDINGS (상세 분석): 커뮤니티에 대한 핵심 인사이트 5~10개.
  - 각 인사이트는 한 줄 요약(`summary`)과 그에 대한 상세 설명(`explanation`)으로 구성됨
  - 모든 설명은 반드시 커뮤니티 데이터에 기반해야 하며, 과장하거나 없는 내용을 추론하지 말 것

결과는 다음과 같은 JSON 포맷으로 출력하세요:
{{
    "title": <제목>,
    "summary": <요약>,
    "rating": <영향력 점수>,
    "rating_explanation": <점수에 대한 설명>,
    "findings": [
        {{
            "summary": <요약1>,
            "explanation": <상세 설명1>
        }},
        {{
            "summary": <요약2>,
            "explanation": <상세 설명2>
        }},
        ...
    ]
}}

# 근거 제시 규칙 (Grounding Rules)
- 커뮤니티에 존재하지 않는 정보는 추가하지 마세요.
- 명확한 증거가 없는 내용은 제거하세요.

# 예시 입력
-----------
텍스트:
```
Entities:
```csv
id,entity,type,description
5,VERDANT OASIS PLAZA,geo,Unity March가 열리는 장소
6,HARMONY ASSEMBLY,organization,Verdant Oasis Plaza에서 행진을 여는 단체
```
Relationships:
```csv
id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza는 Unity March가 열리는 장소
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly는 해당 장소에서 행진을 개최
39,VERDANT OASIS PLAZA,UNITY MARCH,Unity March는 해당 장소에서 진행 중
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight가 이 행진에 대해 보도
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi가 이 장소에서 연설
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly는 Unity March의 조직자
```
```
출력 예시:
{{
    "title": "Unity March 중심의 Verdant Oasis Plaza 커뮤니티",
    "summary": "이 커뮤니티는 Verdant Oasis Plaza를 중심으로 Unity March, Harmony Assembly 등 다양한 엔터티들이 연결되어 있는 구조입니다.",
    "rating": 5.0,
    "rating_explanation": "Unity March가 공공 질서에 미치는 영향 가능성을 고려하여 중간 수준의 영향력으로 평가됨.",
    "findings": [
        {{
            "summary": "Verdant Oasis Plaza는 커뮤니티의 중심 장소임",
            "explanation": "이 장소는 Unity March의 개최지이자 다양한 엔터티들의 연결고리입니다. 이를 통해 공공 질서 및 여론 형성에 중요한 역할을 합니다."
        }},
        {{
            "summary": "Harmony Assembly는 주요 주최자",
            "explanation": "행진의 주관 조직인 Harmony Assembly는 커뮤니티 내 핵심적인 역할을 수행하고 있으며, 이들의 의도와 활동은 지역 사회에 중요한 영향을 미칠 수 있습니다."
        }},
        ...
    ]
}}

# 실제 데이터 사용 시
다음 텍스트에 기반하여 보고서를 작성하세요. 추론하지 말고, 반드시 근거가 있는 내용만 포함하세요.

텍스트:
```
{input_text}
```

보고서는 아래 항목들을 반드시 포함해야 합니다:

- TITLE (제목): 커뮤니티의 대표적인 엔터티 이름을 포함한 간결하고 구체적인 제목
- SUMMARY (요약): 커뮤니티의 전체 구조, 주요 엔터티 간 관계, 핵심 정보에 대한 요약 설명
- IMPACT SEVERITY RATING (영향력 점수): 커뮤니티가 가질 수 있는 영향력 수준 (0~10 사이 소수점 가능)
- RATING EXPLANATION (점수 설명): 위 점수에 대한 한 줄 설명
- DETAILED FINDINGS (상세 분석): 커뮤니티에 대한 핵심 인사이트 5~10개.
  - 각 인사이트는 한 줄 요약(`summary`)과 그에 대한 상세 설명(`explanation`)으로 구성됨
  - 모든 설명은 반드시 커뮤니티 데이터에 기반해야 하며, 과장하거나 없는 내용을 추론하지 말 것

결과는 다음과 같은 JSON 포맷으로 출력하세요:
{{
    "title": <제목>,
    "summary": <요약>,
    "rating": <영향력 점수>,
    "rating_explanation": <점수에 대한 설명>,
    "findings": [
        {{
            "summary": <요약1>,
            "explanation": <상세 설명1>
        }},
        {{
            "summary": <요약2>,
            "explanation": <상세 설명2>
        }},
        ...
    ]
}}

# 근거 제시 규칙 (Grounding Rules)
- 커뮤니티에 존재하지 않는 정보는 추가하지 마세요.
- 명확한 증거가 없는 내용은 제거하세요.


Output:
"""


PROMPTS["entity_extraction"] = """-목표-
다음에 제시된 텍스트 문서와 엔티티 타입 목록을 바탕으로, 해당 타입에 해당하는 엔티티들을 모두 식별하고 이들 간의 명확한 관계를 추출하세요.

-단계-
1. 엔티티 식별:
   문서 내에서 지정된 엔티티 타입({entity_types})에 해당하는 엔티티를 모두 찾아내고, 각각에 대해 다음 정보를 추출합니다:
   - entity_name: 엔티티의 이름 (모두 대문자로 표기)
   - entity_type: 지정된 엔티티 타입 중 하나
   - entity_description: 해당 엔티티의 특성과 활동에 대한 상세 설명

   아래 형식으로 출력합니다:
   ("entity"<|><엔티티_이름><|><엔티티_타입><|><엔티티_설명>)

2. 관계 추출:
   위에서 식별한 엔티티들 중 명확히 관련된 엔티티 쌍(source_entity, target_entity)을 찾고, 각각에 대해 다음 정보를 추출합니다:
   - source_entity: 관계의 출발점이 되는 엔티티 이름
   - target_entity: 관계의 도착점이 되는 엔티티 이름
   - relationship_description: 두 엔티티가 관련 있다고 판단한 이유를 설명하는 문장
   - relationship_strength: 관계의 강도를 나타내는 숫자 (1–10 사이)

   아래 형식으로 출력합니다:
   ("relationship"<|><출발_엔티티><|><도착_엔티티><|><관계_설명><|><관계_강도>)

3. 출력:
   엔티티와 관계 정보를 모두 **하나의 목록**으로 묶어 출력하며, 각 항목은 **{record_delimiter}** 로 구분합니다.
   출력의 마지막은 반드시 {completion_delimiter} 로 마무리해야 합니다.

######################
-예시-
######################
Entity_types: [person, technology, mission, organization, location]
Text:
Alex는 턱을 꽉 깨물며 분노를 참는다. 그의 옆에 있는 Taylor는 권위적인 태도로 모든 걸 지시한다. 그들 사이에는 긴장감이 흐르고, Jordan과의 협업은 마치 그 권위에 대한 조용한 저항처럼 느껴졌다.

Taylor는 뜻밖의 행동을 한다. Jordan 옆에서 멈춰 장비를 조심스럽게 바라보며 말했다. "이 기술이 이해된다면, 모든 것을 바꿀 수 있어요."

그 말은 그들의 신념과도 맞닿아 있었다. Alex는 그 순간을 잊지 못할 것이다.
################
Output:
("entity"<|>"ALEX"<|>"PERSON"<|>"Alex는 팀의 일원이며, Taylor의 권위적인 태도에 반응하며 내부 갈등을 경험하는 인물입니다."){record_delimiter}
("entity"<|>"TAYLOR"<|>"PERSON"<|>"Taylor는 권위적이고 지시적인 리더로 묘사되며, 기술 장비에 대해 경외감을 보이는 변화를 보여줍니다."){record_delimiter}
("entity"<|>"JORDAN"<|>"PERSON"<|>"Jordan은 기술 개발에 헌신적이며 Taylor와의 상호작용에서 중요한 역할을 합니다."){record_delimiter}
("entity"<|>"DEVICE"<|>"TECHNOLOGY"<|>"이 장비는 이야기의 중심 기술이며, 미래를 바꿀 수 있는 잠재력을 가집니다."){record_delimiter}
("relationship"<|>"ALEX"<|>"TAYLOR"<|>"Alex는 Taylor의 권위적인 태도에 영향을 받고 있으며, 그 행동을 관찰합니다."{record_delimiter}7){record_delimiter}
("relationship"<|>"TAYLOR"<|>"JORDAN"<|>"Taylor는 Jordan 옆에서 장비를 관찰하며 잠재적인 협력 관계를 보여줍니다."{record_delimiter}6){record_delimiter}
("relationship"<|>"TAYLOR"<|>"DEVICE"<|>"Taylor는 장비를 경외심을 가지고 바라보며 기술에 대한 인식을 드러냅니다."{record_delimiter}9){completion_delimiter}

######################
-실제 입력-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""




PROMPTS["summarize_entity_descriptions"] = """당신은 주어진 엔티티에 대한 설명들을 종합하여 하나의 통합된 설명을 작성하는 똑똑한 어시스턴트입니다.

## 목표
아래에 주어진 하나 또는 두 개의 엔티티와 그에 대한 설명 목록을 바탕으로, 해당 엔티티에 대한 포괄적이고 일관된 설명을 작성하세요. 모든 설명을 반영하여 정보를 통합해야 하며, 서로 상충되는 설명이 존재할 경우 이를 해소하여 자연스럽고 일관된 문장으로 정리하세요.

- 모든 설명을 세 번째 인칭으로 작성해야 하며,
- 엔티티 이름을 포함하여 어떤 대상을 설명하는지 분명히 해야 합니다.

#######
-데이터-
엔티티: {entity_name}
설명 목록: {description_list}
#######
출력:
"""




PROMPTS["entiti_continue_extraction"] = """이전 추출에서 많은 엔티티가 누락되었습니다. 아래에 동일한 형식으로 빠진 엔티티들을 추가해 주세요:
"""

PROMPTS["entiti_if_loop_extraction"] = """추가로 추출해야 할 엔티티가 남아 있는 것 같습니다. 남아 있다면 YES, 없다면 NO로 답변해 주세요. (YES | NO)
"""

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["local_rag_response"] = """---역할---

당신은 아래에 제공된 테이블 데이터들을 기반으로 사용자의 질문에 응답하는 유능한 어시스턴트입니다.

---목표---

아래 테이블에서 제공된 정보들을 요약하고, 사용자의 질문에 적절하게 응답하는 포맷과 길이로 응답을 생성하세요. 일반적인 지식도 필요하다면 활용해도 좋습니다.

답변은 아래 조건을 반드시 충족해야 합니다:
- 질문과 관련된 테이블의 모든 정보를 요약에 반영할 것
- 정보의 근거가 없는 내용은 포함하지 말 것
- 모르면 "잘 모르겠습니다"라고 대답할 것

---요구 응답 형식---

{response_type}

---데이터 테이블---

{context_data}

---추가 설명---

- 각 섹션에 주제나 항목별 소제목을 포함하세요.
- 마크다운 형식으로 응답을 구성하세요.
"""


PROMPTS["global_map_rag_points"] = """---역할---

당신은 주어진 테이블 데이터를 기반으로 사용자의 질문에 핵심 요점을 정리하여 응답하는 AI 도우미입니다.

---목표---

아래 데이터 테이블들을 기반으로 사용자의 질문에 대한 **핵심 포인트 목록**을 작성하세요.  
입력된 데이터 테이블의 정보를 중심으로 요점을 도출하며,  
내용이 부족하거나 답을 모를 경우 **"모르겠다"고 말하세요. 지어내지 마세요.**

각 핵심 포인트에는 다음 요소들이 포함되어야 합니다:
- **설명(Description)**: 중요한 사실을 설명하는 포괄적인 설명 문장
- **중요도 점수(Importance Score)**: 해당 포인트가 질문에 대해 얼마나 중요한지를 0~100 사이 정수로 점수화한 것  
  (만약 ‘모르겠다’는 식의 답이면 점수는 0이어야 합니다)

응답은 다음과 같은 **JSON 형식**으로 출력되어야 합니다:
{{
    "points": [
        {{"description": "핵심 포인트 1에 대한 설명", "score": 중요도_점수}},
        {{"description": "핵심 포인트 2에 대한 설명", "score": 중요도_점수}}
    ]
}}

모든 설명은 **명확한 근거가 있는 정보에 기반**해야 하며,  
"~일 수 있다", "~일 것이다" 등의 **추측성 표현은 피하거나 명시적으로 표시**하세요.

---입력 데이터 테이블---

{context_data}

---요약---

위 데이터를 바탕으로, 질문에 대한 **핵심 요점들**을 아래 JSON 구조로 응답하세요.

- 허위 정보나 지어낸 내용은 절대 포함하지 마세요.
- 응답이 불가능한 경우, `"points": []` 형태로 빈 리스트를 반환하세요.

응답 포맷:
{{
    "points": [
        {{"description": "핵심 내용 1", "score": 중요도_점수}},
        {{"description": "핵심 내용 2", "score": 중요도_점수}}
    ]
}}
"""


PROMPTS["global_reduce_rag_response"] = """---역할---

당신은 여러 분석가(analyst)가 각각 다른 데이터 조각에 대해 작성한 리포트를 종합하여,  
질문에 대해 최종 요약 응답을 작성하는 AI 분석 도우미입니다.

---목표---

아래에 제공된 여러 분석가들의 리포트를 기반으로 사용자의 질문에 대해 종합적인 응답을 생성하세요.

리포트들은 **중요도 순으로 정렬**되어 있으며,  
중복되거나 불필요한 정보는 제거하고, 핵심적인 포인트를 중심으로 **설명과 해석을 추가**하세요.

- 응답은 사용자가 지정한 길이 및 형식에 맞춰야 합니다.
- 필요시 섹션 구분, 마크다운 스타일링 등을 사용하여 가독성 있게 작성하세요.

다음 사항을 반드시 지키세요:
- **확실한 근거 없는 내용은 절대 포함하지 마세요.**
- LLM이 “~할 것이다” 같은 표현을 쓸 경우, **원문에서 모달 조동사를 유지**하도록 하세요 (예: shall, may, will 등).
- 만약 정보가 부족하면 “답할 수 없음”으로 간단히 응답해도 됩니다.

---목표 응답 형식 및 길이---

{response_type}

---분석가 리포트 (중요도 순)---

{report_data}

---재요약 목표---

위 리포트를 기반으로, 중요한 정보만 남겨 통합된 응답을 작성하세요.  
사용자 질문에 부합하는 **핵심 포인트와 그 의미**를 중심으로 구성하세요.  
불필요한 정보는 모두 제거하세요.

섹션, 제목, 표 형식 등을 자유롭게 활용하세요. 응답은 마크다운 형식을 추천합니다.
"""

PROMPTS["naive_rag_response"] = """당신은 친절한 AI 도우미입니다.

아래는 현재 주어진 배경 지식입니다:
{content_data}
---
이 지식을 바탕으로 사용자의 질문에 답변을 생성하세요.

- 질문에 정확히 답변할 수 없는 경우, **"답변할 수 없습니다"**라고 하세요.
- 사실이 아닌 내용을 **절대 지어내지 마세요.**
- 답변은 주어진 데이터 범위 내에서만 작성하고, 일반적인 상식이나 배경 지식은 필요 시 활용 가능합니다.
- 사용자가 지정한 길이 및 형식({response_type})을 반드시 따르세요.

---응답 시작---
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["default_text_separator"] = [
    # Paragraph separators
    "\n\n",
    "\r\n\r\n",
    # Line breaks
    "\n",
    "\r\n",
    # Sentence ending punctuation
    "。",  # Chinese period
    "．",  # Full-width dot
    ".",  # English period
    "！",  # Chinese exclamation mark
    "!",  # English exclamation mark
    "？",  # Chinese question mark
    "?",  # English question mark
    # Whitespace characters
    " ",  # Space
    "\t",  # Tab
    "\u3000",  # Full-width space
    # Special characters
    "\u200b",  # Zero-width space (used in some Asian languages)
]
