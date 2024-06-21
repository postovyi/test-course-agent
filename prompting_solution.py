import json

from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader

from config import settings
from prompts import prompts 


class CourseAnalyzer:

    def __init__(self):
        self.pdf_path = "custom_data/AI_ML_Course_Description.pdf"
        self.json_path = "custom_data/small_sample.json"
        self.output_path = "selected_individuals.json"
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.gpt_model,
        )
        self.course_name = "Advanced AI and Machine Learning Course"

    def extract_text_from_pdf(self):
        reader = PdfReader(self.pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        return text

    def generate_course_summary(self, course_description):
        prompt = prompts.course_summary_prompt(course_description)
        response = self.llm.invoke(prompt)

        return response.content

    def analyze_profile(self, individual, summary):
        profile_text = f"{individual['job_title']} {individual.get('about', '')} \
                {individual.get('summary', '')} {' '.join(individual.get('posts', []))}"
        prompt = prompts.profile_analysis_prompt(profile_text, summary)
        response = self.llm.invoke(prompt)
        response_body = response.content
        is_interested = response_body.split(".")[0].strip()
        explanation = response_body[len(is_interested) + 1 :].strip()

        return is_interested, explanation

    def create_personalized_message(self, individual):
        prompt = prompts.personalized_message_prompt(individual, self.course_name)
        response = self.llm.invoke(prompt)

        return response.content

    def run(self):
        course_text = self.extract_text_from_pdf()
        summary = self.generate_course_summary(course_text)

        with open(self.json_path, "r") as file:
            individuals = json.load(file)

        selected_individuals = []
        for individual in individuals:
            is_interested, explanation = self.analyze_profile(individual, summary)
            if is_interested == "Yes":
                message = self.create_personalized_message(individual)
                selected_individuals.append(
                    {
                        "name": individual["name"],
                        "job_title": individual["job_title"],
                        "explanation": explanation,
                        "message": message,
                    }
                )

        with open(self.output_path, "w") as outfile:
            json.dump(selected_individuals, outfile, indent=4)


if __name__ == "__main__":
    analyzer = CourseAnalyzer()
    analyzer.run()
