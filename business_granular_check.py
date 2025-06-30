import streamlit as st
from jira import JIRA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
from requests.auth import HTTPBasicAuth
import re

st.set_page_config(page_title="Business Value Assessment AI", layout="wide")
st.title("📊 Business Value Assessment AI")

# --- PROMPTS ---
BUSINESS_VALUE_PROMPT = """
You are a Business Value Analyst Agent. Given a user story or backlog item, along with any context such as goals, risks, deadlines, dependencies, or effort/complexity, your tasks are:

1. Assess the business value of the item considering:
    - Business value or customer impact
    - Deadlines or time sensitivity
    - Dependencies on or by other work
    - Risk of delay or failure
    - Effort or complexity
    - Alignment with strategic goals or company objectives
    - Urgency (regulatory, competitive, or other time-sensitive factors)
    - Potential Return on Investment (ROI)
2. Suggest a **business value score** (High, Medium, Low).
3. Suggest a **priority** (High/Medium/Low or Must-have/Should-have/Nice-to-have) with a brief justification.
4. If important info is missing, state what is needed.

**Input:**  
User Story:  
{user_story}

Context (if any):  
{context}

**Output (format):**
---
**Business Value Assessment:**  
<bullet points for each factor above>

**Business Value Score:** High/Medium/Low

**Priority Suggestion:** Must-have/Should-have/Nice-to-have  
Justification: <your justification>

<If info is missing, mention what's needed>
---
"""

GRANULARITY_PROMPT = """
You are a requirements analyst. Given the following user story, decide if it is “granular” (i.e., focused, specific, and can be completed within a single sprint).

Reply only with 'Yes' if the story is granular, or 'No' if it is not.

User Story:
{user_story}
"""

def clear_connection_state():
    for k in [
        "jira_host", "jira_email", "jira_api_token", "jira_project_key",
        "connected", "custom_field_id", "last_assessment",
        "last_selected_issue_key"
    ]:
        if k in st.session_state:
            del st.session_state[k]

# --- Disconnect Button ---
if st.session_state.get("connected", False):
    colc, cold = st.columns([10, 1])
    with cold:
        if st.button("Disconnect"):
            clear_connection_state()
            st.rerun()

# --- Connection Form ---
if not st.session_state.get("connected", False):
    st.subheader("Connect to Jira")
    with st.form("connection_form"):
        jira_host = st.text_input("Jira Host URL (e.g. https://yourdomain.atlassian.net)", value=st.session_state.get("jira_host", ""))
        jira_email = st.text_input("Jira Email", value=st.session_state.get("jira_email", ""))
        jira_api_token = st.text_input("Jira API Token", type="password", value=st.session_state.get("jira_api_token", ""))
        jira_project_key = st.text_input("Jira Project Key", value=st.session_state.get("jira_project_key", ""))
        submitted = st.form_submit_button("Connect")

    if submitted:
        if not (jira_host and jira_email and jira_api_token and jira_project_key):
            st.warning("Please fill in all fields to connect.")
        else:
            st.session_state["jira_host"] = jira_host.strip()
            st.session_state["jira_email"] = jira_email.strip()
            st.session_state["jira_api_token"] = jira_api_token.strip()
            st.session_state["jira_project_key"] = jira_project_key.strip()
            try:
                jira = JIRA(server=jira_host, basic_auth=(jira_email, jira_api_token))
                st.session_state["connected"] = True
                st.success(f"Connected as {jira_email} to JIRA: {jira_project_key}")
            except Exception as e:
                st.session_state["connected"] = False
                st.error(f"Failed to connect to Jira: {e}")

def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])

def check_granularity(user_story):
    chain = LLMChain(
        llm=get_llm(),
        prompt=PromptTemplate.from_template(GRANULARITY_PROMPT)
    )
    response = chain.run({"user_story": user_story})
    return response.strip().lower().startswith('yes')

def extract_refined_story(description_text):
    """
    Extract the actual user story from a markdown block produced by the refiner.
    More robust against newlines, whitespace, or extra formatting.
    """
    if not description_text:
        return ""
    # Try strict match first
    match = re.search(r"\*\*Refined User Story:\*\*\s*(.+)", description_text)
    if match:
        # If there's a new line before "**Acceptance Criteria:**", cut at that point
        story_line = match.group(1).strip()
        # Sometimes the story may spill to next line; try to capture just the first line
        story_line = story_line.split('\n')[0].strip()
        return story_line
    # Try for "As a ..." type sentences as a fallback
    match = re.search(r"(As a .+? so that .+?)(?:\n|$)", description_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return description_text.strip()

# --- If connected, main UI ---
if st.session_state.get("connected", False):
    jira_host = st.session_state["jira_host"]
    jira_email = st.session_state["jira_email"]
    jira_api_token = st.session_state["jira_api_token"]
    jira_project_key = st.session_state["jira_project_key"]

    # --- Custom Field Creation/Retrieval ---
    def get_custom_field_id(field_name):
        url = f"{jira_host}/rest/api/3/field"
        auth = HTTPBasicAuth(jira_email, jira_api_token)
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers, auth=auth)
        if response.status_code == 200:
            fields = response.json()
            for field in fields:
                if field['name'] == field_name:
                    return field['id']
        return None

    def create_custom_field(field_name, description, field_type="com.atlassian.jira.plugin.system.customfieldtypes:textarea"):
        url = f"{jira_host}/rest/api/3/field"
        auth = HTTPBasicAuth(jira_email, jira_api_token)
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        payload = {
            "name": field_name,
            "description": description,
            "type": field_type,
            "searcherKey": "textsearcher"
        }
        response = requests.post(url, json=payload, headers=headers, auth=auth)
        if response.status_code == 201:
            return response.json().get('id')
        elif response.status_code == 400 and "A custom field with this name already exists" in response.text:
            return None
        else:
            return None

    FIELD_NAME = "Business Value"
    FIELD_DESCRIPTION = "Business Value assessment generated by AI."
    custom_field_id = get_custom_field_id(FIELD_NAME)
    custom_field_status = ""

    if not custom_field_id:
        new_field_id = create_custom_field(FIELD_NAME, FIELD_DESCRIPTION)
        if new_field_id:
            st.success(f"Custom field '{FIELD_NAME}' created in Jira.")
            custom_field_id = new_field_id
        else:
            custom_field_id = get_custom_field_id(FIELD_NAME)
            if custom_field_id:
                st.info(f"Custom field '{FIELD_NAME}' already exists in Jira.")
            else:
                st.error(f"Could not create or find the custom field '{FIELD_NAME}'.")
    else:
        st.info(f"Custom field '{FIELD_NAME}' found in Jira.")

    st.session_state["custom_field_id"] = custom_field_id

    # --- BV Priority Extraction & Sort ---
    def extract_bv_score(bv_field_val):
        if not bv_field_val:
            return None
        match = re.search(r"\*\*Business Value Score:\*\*\s*(High|Medium|Low)", bv_field_val, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
        return None

    def bv_score_to_int(score):
        return {"High": 3, "Medium": 2, "Low": 1}.get(score, 0)

    # --- Jira Instance ---
    try:
        jira = JIRA(server=jira_host, basic_auth=(jira_email, jira_api_token))
    except Exception as e:
        st.error(f"Failed to connect to Jira after authentication: {e}")
        st.stop()

    # --- Fetch Issues ---
    try:
        jql = f'project={jira_project_key} ORDER BY created ASC'
        issues = jira.search_issues(jql, maxResults=30)
    except Exception as e:
        st.error(f"Failed to load issues: {e}")
        issues = []

    if issues and custom_field_id:
        show_only_unassessed = st.checkbox("Show only stories without Business Value", value=False)
        # --- Collect and sort issues by Business Value Score ---
        issues_with_scores = []
        for i in issues:
            bv_content = getattr(i.fields, custom_field_id, None)
            bv_score = extract_bv_score(bv_content)
            is_unassessed = not bv_score
            if show_only_unassessed and not is_unassessed:
                continue
            issues_with_scores.append((i, bv_score_to_int(bv_score), bv_score))

        # Sort by numerical BV score descending, then by issue key for tie-breaker
        issues_with_scores.sort(key=lambda tup: (-tup[1], tup[0].key))

        issue_titles = []
        filtered_issues = []
        for i, score, bv_label in issues_with_scores:
            label = f"{'⭐️ ' if score==3 else ''}{i.key}: {i.fields.summary}"
            if bv_label:
                label += f"  (BV: {bv_label})"
            issue_titles.append(label)
            filtered_issues.append(i)

        if not issue_titles:
            st.warning("No matching stories found.")
            st.stop()

        selected = st.selectbox(
            "Select a user story for business value assessment (sorted by Business Value):",
            issue_titles
        )
        selected_issue = filtered_issues[issue_titles.index(selected)]
        summary = selected_issue.fields.summary
        description = selected_issue.fields.description or ""

        # --- EXTRACT ONLY THE USER STORY FOR GRANULARITY CHECK ---
        refined_user_story = extract_refined_story(description)
        # Fallback to summary if extraction fails
        story_for_granularity = refined_user_story if refined_user_story else summary

        # DEBUG: Show what will be checked for granularity!
        st.markdown("#### 🛠️ Text sent for Granularity Check:")
        st.code(story_for_granularity)

        st.subheader("Granularity Check")
        with st.spinner("Checking if user story is granular..."):
            is_granular = check_granularity(story_for_granularity)

        if not is_granular:
            st.warning(
                "🚩 This user story is **not granular** (i.e., not focused or small enough for a single sprint). "
                "Please refine this story in the **User Story Refiner** app before assessing business value."
            )
            st.stop()
        else:
            st.success("✅ This user story is granular. Proceed to business value assessment.")

        # ----------- BV Assessment UI --------------
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 Original Story")
            st.markdown(f"**Summary:** {summary}")
            st.markdown(f"**Description:** {description}")

        with col2:
            st.subheader("💡 Business Value Assessment")
            with st.form("assessment_form", clear_on_submit=True):
                context = st.text_area("Additional Context (optional)", value="")
                submitted = st.form_submit_button("🔍 Assess Business Value")
                if submitted:
                    with st.spinner("Assessing with AI..."):
                        llm = ChatOpenAI(
                            model="gpt-4o",
                            api_key=st.secrets["OPENAI_API_KEY"],
                            temperature=0.2,
                            max_tokens=1024
                        )
                        chain = LLMChain(
                            llm=llm,
                            prompt=PromptTemplate.from_template(BUSINESS_VALUE_PROMPT)
                        )
                        try:
                            assessment = chain.run({"user_story": f"{summary}\n\n{description}", "context": context})
                        except Exception as e:
                            st.error(f"OpenAI Error: {e}")
                            assessment = ""
                        if assessment:
                            st.markdown(f"**Business Value Assessment Output:**\n\n{assessment}")
                            st.session_state["last_assessment"] = assessment
                            st.session_state["last_selected_issue_key"] = selected_issue.key

            # Show Update Jira if an assessment is present for this story
            if (
                st.session_state.get("last_assessment")
                and st.session_state.get("last_selected_issue_key") == selected_issue.key
            ):
                if st.button("📌 Update Jira with Business Value", key="update_jira_btn"):
                    update_fields = {custom_field_id: st.session_state["last_assessment"]}
                    try:
                        jira.issue(selected_issue.key).update(fields=update_fields)
                        st.success(f"Business Value updated for {selected_issue.key} in Jira!")
                    except Exception as e:
                        st.error(f"Failed to update Jira: {e}")

    else:
        st.warning("No issues found in the selected project or custom field is missing.")
