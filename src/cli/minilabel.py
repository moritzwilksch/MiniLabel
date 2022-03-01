from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.align import Align
from rich.markdown import Markdown
from rich.style import Style
from rich.text import Text
from src.labeling_manager import LabelingManager, MongoConnector
import os
import yaml

with open("src/cli/task_config.yaml") as f:
    CONFIG = yaml.safe_load(f)


class LabelingCLI:
    def __init__(
        self, db_connector: MongoConnector, manager: LabelingManager, c: Console
    ) -> None:
        self.db_connector = db_connector
        self.manager = manager
        self.c = c

    def homescreen(self):
        self.c.clear()
        self.c.print(Markdown("# MiniLabel"))
        self.c.print("[bold underline]Options[/]")
        self.c.print("- 🚀 (1) Start labeling")
        self.c.print("- 📊 (2) Show stats")
        self.c.print("- 💿 (3) Show DB connection")
        self.c.print()
        option = Prompt.ask("Choose option", choices=["1", "2", "3", "q"])
        self.c.clear()
        return option

    def start_labeling(self):
        legend = "  ".join(f"{x['number']}...{x['title']}" for x in CONFIG.get("labels"))
        number_label_mapping = {
            e.get("number"): e.get("title") for e in CONFIG.get("labels")
        }

        while True:
            self.c.clear()

            try:
                sample = self.manager.get_sample()
            except RuntimeError:
                self.c.print("No more data to sample.")
                _ = Prompt.ask("Press enter to continue")
                break

            p = Panel(
                sample.get("content"),
                title=Text(f"id = {sample['_id']}"),
                border_style=Style(dim=True),
            )
            self.c.print(p)
            self.c.print("\n" * 2)
            self.c.print(Align(self.manager.status_as_string(), align="center"))
            self.c.print(
                Align("[grey]" + legend + "[/]", align="center"), style=Style(dim=True)
            )

            choice = Prompt.ask("Choose", choices=["1", "2", "3", "q"])
            if choice == "q":
                break
            else:
                self.manager.update_one(
                    sample["_id"], number_label_mapping.get(int(choice))
                )

    def run(self):
        while True:
            option = self.homescreen()

            if option == "q":
                exit(0)
            if option == "1":
                self.start_labeling()

            elif option == "2":
                self.c.print(self.manager.get_status())
                _ = Prompt.ask("Press enter to continue")

            elif option == "3":
                print(
                    f"MongoDB connection: user = {user[:5]}..., password = {password[:5]}..."
                )
                _ = Prompt.ask("Press enter to continue")


if __name__ == "__main__":
    user = os.getenv("MONGO_INITDB_ROOT_USERNAME")
    password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
    conn = MongoConnector(
        user,
        password,
        host="157.90.167.200",
        port=27017,
        db="data_labeling",
        collection="dev_coll",
    )

    manager = LabelingManager(db_connector=conn, model=None)
    c = Console()
    cli = LabelingCLI(conn, manager, c)
    cli.run()
