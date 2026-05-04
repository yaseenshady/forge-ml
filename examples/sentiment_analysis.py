"""
Example: classify sentiment of movie reviews.

Run:
    forge run "classify sentiment of movie reviews into positive or negative"
"""
import asyncio
from forge_ml.orchestrator import Orchestrator

async def main():
    orc = Orchestrator()
    ctx = await orc.run("classify sentiment of movie reviews into positive or negative")
    print(f"Dataset: {ctx.dataset_name}")
    print(f"Cost: ${ctx.total_cost_usd:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
