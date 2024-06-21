class Prompts:

    @staticmethod
    def course_summary_prompt(course_description):
        return f"Write the summary for given course description:\\n\\n{course_description}"

    @staticmethod
    def profile_analysis_prompt(profile_text, summary):
        return f"Based on the course summary {summary}, determine if the following profile is interested in this course:\\n\\n{profile_text}\\n\\nThe first sentence must be strictly only Yes or No indicating if the person fits. Then provide a brief explanation why."

    @staticmethod
    def personalized_message_prompt(individual, course_name):
        return f"Create a personalized formal letter for {individual['name']} to invite them to the {course_name} based on their profile: {individual}"

prompts = Prompts()