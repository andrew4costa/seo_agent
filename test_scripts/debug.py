import sys
import asyncio

async def main():
    print("====== DEBUG SCRIPT STARTING ======")
    try:
        # Import the main function from the specified file 
        module_name = sys.argv[1] if len(sys.argv) > 1 else 'seo_fixed.py'
        module_name = module_name.replace('.py', '')
        
        print(f"Importing main function from {module_name}")
        module = __import__(module_name)
        main_func = module.main
        
        # Set command line args programmatically
        sys.argv = [module_name + '.py', 'https://example.com', '--limit', '1']
        print(f"Running with args: {sys.argv}")
        
        # Run the main function
        await main_func()
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("====== DEBUG SCRIPT FINISHED ======")

if __name__ == "__main__":
    asyncio.run(main()) 