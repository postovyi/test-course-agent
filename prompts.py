class Prompts:
    """Class containing prompts for course recommendations"""

    @staticmethod
    def course_summary_prompt(course_description: str) -> str:
        """Prompt for course description summary generation.

        Args:
            course_description (str): course description stored as a string;

        Returns:
            str: the prompt itself.
        """
        return (
            f"Write the summary for given course description:\\n\\n{course_description}"
        )

    @staticmethod
    def profile_analysis_prompt(profile_text: str, summary: str) -> str:
        """Prompt for user profile analysis;
           used to determine whether the user fits for the course or not.

        Args:
            profile_text (str): user profile description;
            summary (str): course summary.

        Returns:
            str: the prompt itself.
        """
        return f"Based on the course summary {summary}, determine if the following profile is interested in this course:\\n\\n{profile_text}\\n\\nThe first sentence must be strictly only Yes or No indicating if the person fits. Then provide a brief explanation why."

    @staticmethod
    def personalized_message_prompt(individual: str, course_name: str) -> str:
        """Prompt for personalized message generation.

        Args:
            individual (str): name of the user to recommend the course for;
            course_name (str): name of the course to recommend.

        Returns:
            str: the prompt itself.
        """
        return f"Craft a personalized message to {individual['name']} to invite them to the {course_name} course based on their profile: {individual}"


prompts = Prompts()
