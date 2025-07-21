import base64
from PIL import Image
import io
import os
from datetime import datetime

def convert_base64_to_image(base64_string, output_filename=None):
    """
    Convert base64 string to image file
    
    Args:
        base64_string (str): Base64 encoded image data
        output_filename (str): Optional output filename. If None, generates timestamp-based name
    
    Returns:
        str: Path to saved image file
    """
    try:
        # Clean the base64 string (remove data URL prefix if present)
        original_length = len(base64_string)
        print(f"🔍 Original base64 length: {original_length}")
        
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
            print("🧹 Removed data URL prefix")
        
        # Remove any whitespace/newlines/quotes
        base64_string = base64_string.strip().replace('\n', '').replace('\r', '').replace('"', '').replace("'", '')
        print(f"🧹 Cleaned base64 length: {len(base64_string)}")
        
        # Check if base64 string looks valid
        if len(base64_string) < 100:
            print("❌ Base64 string too short to be a valid image")
            return None
        
        # Add padding if needed
        missing_padding = len(base64_string) % 4
        if missing_padding:
            base64_string += '=' * (4 - missing_padding)
            print(f"🔧 Added {4 - missing_padding} padding characters")
        
        # Try to decode base64
        try:
            image_bytes = base64.b64decode(base64_string)
            print(f"✅ Successfully decoded base64 to {len(image_bytes)} bytes")
        except Exception as decode_error:
            print(f"❌ Base64 decode error: {decode_error}")
            return None
        
        # Check if decoded bytes look like image data
        if len(image_bytes) < 1000:
            print("❌ Decoded bytes too small to be an image")
            return None
        
        # Print first few bytes to debug
        print(f"🔍 First 20 bytes: {image_bytes[:20]}")
        
        # Try to identify image format from bytes
        image_signatures = {
            b'\x89PNG\r\n\x1a\n': 'PNG',
            b'\xff\xd8\xff': 'JPEG',
            b'GIF87a': 'GIF',
            b'GIF89a': 'GIF',
            b'RIFF': 'WEBP',
        }
        
        detected_format = None
        for signature, format_name in image_signatures.items():
            if image_bytes.startswith(signature):
                detected_format = format_name
                break
        
        if detected_format:
            print(f"🎨 Detected image format: {detected_format}")
        else:
            print("⚠️  Could not detect image format from bytes")
        
        # Try to create PIL Image from bytes
        try:
            image = Image.open(io.BytesIO(image_bytes))
            print(f"✅ Successfully opened image with PIL")
        except Exception as pil_error:
            print(f"❌ PIL error: {pil_error}")
            
            # Try saving as raw bytes to debug
            debug_filename = "debug_raw_bytes.bin"
            with open(debug_filename, 'wb') as f:
                f.write(image_bytes)
            print(f"🔧 Saved raw bytes to {debug_filename} for debugging")
            
            return None
        
        # Generate filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            format_ext = {
                'PNG': 'png',
                'JPEG': 'jpg',
                'JPG': 'jpg',
                'WEBP': 'webp',
                'GIF': 'gif'
            }
            ext = format_ext.get(image.format, 'png')
            output_filename = f"converted_image_{timestamp}.{ext}"
        elif not output_filename.endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
            # Add extension if not provided
            ext = 'png'
            if image.format:
                format_ext = {
                    'PNG': 'png',
                    'JPEG': 'jpg',
                    'JPG': 'jpg',
                    'WEBP': 'webp',
                    'GIF': 'gif'
                }
                ext = format_ext.get(image.format, 'png')
            output_filename = f"{output_filename}.{ext}"
        
        # Save the image
        image.save(output_filename)
        
        print(f"✅ Image saved successfully as: {output_filename}")
        print(f"📏 Image dimensions: {image.width}x{image.height}")
        print(f"🎨 Image format: {image.format}")
        print(f"📁 Full path: {os.path.abspath(output_filename)}")
        
        return output_filename
        
    except Exception as e:
        print(f"❌ Error converting base64 to image: {e}")
        print(f"🔧 Try checking if your base64 string is valid")
        return None

def convert_from_file(file_path, output_filename=None):
    """
    Read base64 from file and convert to image
    
    Args:
        file_path (str): Path to text file containing base64 data
        output_filename (str): Optional output filename
    
    Returns:
        str: Path to saved image file
    """
    try:
        # Read base64 data from file
        with open(file_path, 'r', encoding='utf-8') as file:
            base64_string = file.read()
        
        print(f"📖 Reading base64 data from: {file_path}")
        print(f"📊 Base64 string length: {len(base64_string)} characters")
        
        # Convert to image
        return convert_base64_to_image(base64_string, output_filename)
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def batch_convert_from_file(file_path, output_directory="converted_images"):
    """
    Convert multiple base64 strings from a file (one per line or separated by specific delimiter)
    
    Args:
        file_path (str): Path to text file containing base64 data
        output_directory (str): Directory to save converted images
    
    Returns:
        list: List of saved image file paths
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Try to split by common delimiters
        possible_delimiters = ['\n\n', '---', '===', '\n']
        base64_strings = []
        
        for delimiter in possible_delimiters:
            if delimiter in content:
                base64_strings = [s.strip() for s in content.split(delimiter) if s.strip()]
                break
        
        if not base64_strings:
            # Treat entire content as single base64 string
            base64_strings = [content.strip()]
        
        print(f"📖 Found {len(base64_strings)} base64 string(s) in file")
        
        saved_files = []
        for i, base64_string in enumerate(base64_strings, 1):
            if len(base64_string) > 100:  # Only process if it looks like valid base64
                output_filename = os.path.join(output_directory, f"image_{i}.png")
                result = convert_base64_to_image(base64_string, output_filename)
                if result:
                    saved_files.append(result)
        
        print(f"✅ Successfully converted {len(saved_files)} image(s)")
        return saved_files
        
    except Exception as e:
        print(f"❌ Error in batch conversion: {e}")
        return []

# Example usage and main execution
if __name__ == "__main__":
    import sys
    
    print("🖼️  Base64 to Image Converter")
    print("=" * 40)
    
    # Check if file path provided as command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"Converting from file: {file_path}")
        result = convert_from_file(file_path, output_file)
        
        if result:
            print(f"\n🎉 Conversion completed successfully!")
        else:
            print(f"\n💥 Conversion failed!")
    else:
        # Interactive mode
        print("\nChoose an option:")
        print("1. Convert from file (single image)")
        print("2. Batch convert from file (multiple images)")
        print("3. Convert base64 string directly")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            file_path = input("Enter path to text file containing base64 data: ").strip()
            output_file = input("Enter output filename (press Enter for auto-generated): ").strip()
            output_file = output_file if output_file else None
            
            result = convert_from_file(file_path, output_file)
            
        elif choice == "2":
            file_path = input("Enter path to text file containing base64 data: ").strip()
            output_dir = input("Enter output directory (press Enter for 'converted_images'): ").strip()
            output_dir = output_dir if output_dir else "converted_images"
            
            results = batch_convert_from_file(file_path, output_dir)
            
        elif choice == "3":
            print("Paste your base64 string (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            
            base64_string = "\n".join(lines)
            output_file = input("Enter output filename (press Enter for auto-generated): ").strip()
            output_file = output_file if output_file else None
            
            result = convert_base64_to_image(base64_string, output_file)
            
        else:
            print("Invalid choice!")

# Quick functions for direct use
def quick_convert(file_path):
    """Quick conversion function"""
    return convert_from_file(file_path)

def quick_batch_convert(file_path):
    """Quick batch conversion function"""
    return batch_convert_from_file(file_path)