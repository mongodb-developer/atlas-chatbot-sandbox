# Atlas Chatbot Sandbox

This repository contains a chatbot project.

## Files

- **chatbot.py**: Main chatbot script.
- **requirements.txt**: Python dependencies.

## Prerequisites

The deployment require a docker deamon. The most convenient way is to use GitHub [codespaces](https://codespaces.new/mongodb-developer/atlas-chatbot-sandbox).

## Getting Started

1. Clone the repository:
   ```sh
   git clone https://github.com/mongodb-developer/atlas-chatbot-sandbox.git
   ```
2. Navigate to the project directory:
   ```sh
   cd atlas-chatbot-sandbox
   ```
3. Ensure Docker is running on your machine.
4. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```
5. Run the chatbot:
   ```sh
   chainlit run chatbot.py
   ```

## Local Atlas Deployment

This project uses the [Tomodo](https://tomodo.dev) project to have a local Atlas deployment for vector search.

## License

This project is licensed under the MIT License.
