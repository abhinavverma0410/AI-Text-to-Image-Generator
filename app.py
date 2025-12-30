import torch
import base64
import io
import time
import random
from pathlib import Path
from datetime import datetime
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)
server = app.server

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
print(f"🔧 Device: {device}")
print(f"📊 CUDA Available: {CUDA_AVAILABLE}")

# Global pipeline
pipe = None
is_generating = False

# Auto negative prompts
AUTO_NEGATIVE = (
    "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, "
    "out of frame, extra limbs, disfigured, deformed, body out of frame, "
    "bad anatomy, watermark, signature, cut off, low contrast, underexposed, "
    "overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy"
)

def load_model_fixed():
    """Load model with proper CUDA handling"""
    global pipe
    try:
        print("🔄 Loading Stable Diffusion model...")
        
        # Determine dtype
        dtype = torch.float16 if CUDA_AVAILABLE else torch.float32
        print(f"   Using precision: {dtype}")
        
        # Load pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Move to device
        pipe = pipe.to(device)
        print(f"   Model moved to {device}")
        
        # Enable optimizations
        if CUDA_AVAILABLE:
            pipe.enable_attention_slicing()
            print("   ✓ Attention slicing enabled")
            
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("   ✓ XFormers enabled")
            except:
                print("   ℹ️ XFormers not available")
            
            # Use faster scheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True
            )
            print("   ✓ Fast scheduler loaded")
        
        # Warm up the model
        print("   Warming up model...")
        with torch.no_grad():
            # Create generator on CPU (FIXED)
            generator = torch.Generator(device="cpu").manual_seed(42)
            
            if CUDA_AVAILABLE:
                with torch.amp.autocast(device_type="cuda"):
                    _ = pipe(
                        prompt="warmup",
                        negative_prompt=AUTO_NEGATIVE,
                        num_inference_steps=1,
                        height=64,
                        width=64,
                        generator=generator,
                    )
            else:
                _ = pipe(
                    prompt="warmup",
                    negative_prompt=AUTO_NEGATIVE,
                    num_inference_steps=1,
                    height=64,
                    width=64,
                    generator=generator,
                )
        
        print("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_image_fixed(prompt, steps=25, guidance=7.5, size=512, seed=None):
    """Generate image with FIXED CUDA generator"""
    try:
        start_time = time.time()
        
        # Use provided seed or generate random
        if seed is None or seed == 0:
            seed = random.randint(1, 2147483647)
        
        # FIXED: Generator must be on CPU
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        # Enhance prompt
        enhanced_prompt = f"{prompt}, high quality, detailed, masterpiece, 8k"
        
        # Generate
        with torch.no_grad():
            if CUDA_AVAILABLE:
                # Use autocast for mixed precision on CUDA
                with torch.amp.autocast(device_type="cuda"):
                    image = pipe(
                        prompt=enhanced_prompt,
                        negative_prompt=AUTO_NEGATIVE,
                        height=size,
                        width=size,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        generator=generator,  # Generator on CPU is fine
                    ).images[0]
            else:
                # CPU generation
                image = pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=AUTO_NEGATIVE,
                    height=size,
                    width=size,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                ).images[0]
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{safe_prompt}_{timestamp}.png"
        
        save_dir = Path("generated_images")
        save_dir.mkdir(exist_ok=True)
        image.save(save_dir / filename)
        
        gen_time = time.time() - start_time
        
        print(f"✅ Generated '{prompt[:30]}...' in {gen_time:.1f}s")
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{img_str}",
            "filename": filename,
            "time": gen_time,
            "seed": seed
        }
        
    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🚀 AI Image Generator", 
                style={'textAlign': 'center', 'color': 'white', 'marginTop': '20px'}),
        html.Div([
            html.Span("⚡ Fast Generation", className="badge bg-success me-2"),
            html.Span(f"Running on: {device.type.upper()}", 
                        className="badge bg-info me-2"),
            html.Span("No Negative Prompts Needed", className="badge bg-warning")
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    ], style={'background': 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)', 
                'padding': '20px', 'borderRadius': '10px', 'marginBottom': '30px'}),
    
    # Main Content
    dbc.Container([
        dbc.Row([
            # Left Panel - Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🎨 Create Your Image"),
                    dbc.CardBody([
                        html.Label("Describe what you want to create:", 
                                    style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                        dcc.Textarea(
                            id='prompt-input',
                            value='A beautiful fantasy landscape with mountains and waterfall',
                            rows=4,
                            placeholder='Be descriptive for best results...',
                            style={
                                'width': '100%',
                                'padding': '15px',
                                'borderRadius': '8px',
                                'border': '1px solid #ddd',
                                'fontSize': '16px',
                                'marginBottom': '20px'
                            }
                        ),
                        
                        html.Div([
                            html.Label("Quality:", style={'fontWeight': 'bold'}),
                            dcc.Slider(
                                id='steps-slider',
                                min=20, max=50, step=5,
                                value=30,
                                marks={20: 'Fast', 30: 'Good', 40: 'Better', 50: 'Best'},
                                tooltip={'placement': 'bottom'}
                            )
                        ], style={'marginBottom': '20px'}),
                        
                        html.Div([
                            html.Label("Image Size:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                            dbc.RadioItems(
                                id='size-select',
                                options=[
                                    {'label': '512x512 (Fast)', 'value': 512},
                                    {'label': '768x768 (Balanced)', 'value': 768},
                                    {'label': '1024x1024 (Detailed)', 'value': 1024},
                                ],
                                value=512,
                                inline=True
                            )
                        ], style={'marginBottom': '20px'}),
                        
                        dbc.Button(
                            '⚡ Generate Image',
                            id='generate-btn',
                            color='primary',
                            size='lg',
                            className='w-100',
                            style={'fontSize': '18px', 'padding': '15px'}
                        ),
                        
                        html.Div(id='model-status', className='mt-3')
                    ])
                ]),
                
                # Quick Examples
                dbc.Card([
                    dbc.CardHeader("💡 Quick Examples"),
                    dbc.CardBody([
                        dbc.ButtonGroup([
                            dbc.Button("🐉 Dragon", id='ex-1', color='outline-secondary', size='sm', className='m-1'),
                            dbc.Button("🌃 City", id='ex-2', color='outline-secondary', size='sm', className='m-1'),
                            dbc.Button("🏰 Castle", id='ex-3', color='outline-secondary', size='sm', className='m-1'),
                            dbc.Button("🚀 Space", id='ex-4', color='outline-secondary', size='sm', className='m-1'),
                        ], vertical=False, className='w-100')
                    ])
                ], className='mt-3')
            ], md=4),
            
            # Right Panel - Results
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "🖼️ Generated Image",
                        html.Span(id='gen-time', className='badge bg-info float-end')
                    ]),
                    dbc.CardBody([
                        # Progress Bar
                        html.Div([
                            dbc.Progress(
                                id='progress-bar',
                                value=0,
                                striped=True,
                                animated=True,
                                style={'height': '10px'}
                            ),
                            html.Div(id='progress-text', className='text-center mt-2 small')
                        ], id='progress-container', style={'display': 'none'}),
                        
                        # Image Display
                        html.Div([
                            html.Img(
                                id='generated-image',
                                className='img-fluid rounded',
                                style={'display': 'none', 'maxHeight': '70vh'}
                            ),
                            html.Div(
                                id='placeholder',
                                className='text-center py-5',
                                children=[
                                    html.I(className='fas fa-image fa-4x text-muted mb-3'),
                                    html.H5('Your image will appear here', className='text-muted'),
                                    html.P('Describe your vision and click Generate', 
                                            className='text-muted small')
                                ]
                            )
                        ]),
                        
                        # Download Button
                        html.Div([
                            dbc.Button(
                                '💾 Download Image',
                                id='download-btn',
                                color='success',
                                className='w-100 mt-3',
                                style={'display': 'none'}
                            ),
                            dcc.Download(id='download-file')
                        ])
                    ], style={'minHeight': '500px'})
                ])
            ], md=8)
        ])
    ], fluid=True),
    
    # Hidden elements
    dcc.Store(id='image-data'),
    dcc.Interval(id='progress-interval', interval=300, disabled=True),
    dcc.Interval(id='init-interval', interval=1000, n_intervals=0, max_intervals=5)
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f5f5f5', 'minHeight': '100vh'})


@app.callback(
    Output('model-status', 'children'),
    Input('init-interval', 'n_intervals')
)
def init_model(n):
    """Initialize model on startup"""
    if n == 0:
        return dbc.Alert("Loading AI model...", color='info')
    
    if n == 1:
        success = load_model_fixed()
        if success:
            return dbc.Alert([
                html.I(className='fas fa-check-circle me-2'),
                "✅ Model loaded and ready!"
            ], color='success')
        else:
            return dbc.Alert([
                html.I(className='fas fa-exclamation-triangle me-2'),
                "❌ Failed to load model"
            ], color='danger')
    
    raise PreventUpdate

@app.callback(
    [
        Output('progress-container', 'style'),
        Output('progress-bar', 'value'),
        Output('progress-text', 'children'),
        Output('progress-interval', 'disabled')
    ],
    Input('generate-btn', 'n_clicks'),
    prevent_initial_call=True
)
def start_generation(n_clicks):
    """Show progress bar when generation starts"""
    if n_clicks:
        return (
            {'display': 'block', 'marginBottom': '20px'},
            10,
            '🚀 Starting image generation...',
            False  # Enable interval
        )
    raise PreventUpdate

@app.callback(
    [
        Output('progress-bar', 'value', allow_duplicate=True),
        Output('progress-text', 'children', allow_duplicate=True)
    ],
    Input('progress-interval', 'n_intervals'),
    prevent_initial_call=True
)
def update_progress(n):
    """Update progress bar animation"""
    # Animate progress from 10% to 90%
    progress = min(10 + (n * 8), 90)
    
    if progress < 30:
        text = "🎨 Initializing AI model..."
    elif progress < 60:
        text = "✨ Creating your image..."
    else:
        text = "⚡ Adding final details..."
    
    return progress, text

@app.callback(
    [
        Output('generated-image', 'src'),
        Output('generated-image', 'style'),
        Output('placeholder', 'style'),
        Output('download-btn', 'style'),
        Output('gen-time', 'children'),
        Output('progress-container', 'style', allow_duplicate=True),
        Output('image-data', 'data'),
        Output('progress-interval', 'disabled', allow_duplicate=True)
    ],
    Input('generate-btn', 'n_clicks'),
    [
        State('prompt-input', 'value'),
        State('steps-slider', 'value'),
        State('size-select', 'value')],
    prevent_initial_call=True,
    running=[
        (Output('generate-btn', 'disabled'), True, False),
        (Output('progress-interval', 'disabled'), False, True),
    ]
)
def generate_and_display(n_clicks, prompt, steps, size):
    """Generate image and update display"""
    if not n_clicks or not prompt:
        raise PreventUpdate
    
    # Generate image
    result = generate_image_fixed(prompt, steps=steps, size=size)
    
    if not result["success"]:
        # Show error
        return (
            '',
            {'display': 'none'},
            {'display': 'block'},
            {'display': 'none'},
            'Error',
            {'display': 'none'},
            None,
            True
        )
    
    # Update UI with result
    time_text = f"{result['time']:.1f}s"
    
    return (
        result["image"],
        {'display': 'block', 'maxHeight': '70vh', 'width': 'auto'},
        {'display': 'none'},
        {'display': 'block'},
        time_text,
        {'display': 'none'},
        {'src': result["image"], 'filename': result["filename"]},
        True
    )

@app.callback(
    Output('download-file', 'data'),
    Input('download-btn', 'n_clicks'),
    State('image-data', 'data'),
    prevent_initial_call=True
)
def download_image(n_clicks, data):
    """Download generated image"""
    if n_clicks and data:
        img_bytes = base64.b64decode(data['src'].split(',')[1])
        return dcc.send_bytes(img_bytes, data['filename'])
    raise PreventUpdate

@app.callback(
    Output('prompt-input', 'value'),
    [
        Input('ex-1', 'n_clicks'),
        Input('ex-2', 'n_clicks'),
        Input('ex-3', 'n_clicks'),
        Input('ex-4', 'n_clicks')
    ],
    prevent_initial_call=True
)
def load_example(ex1, ex2, ex3, ex4):
    """Load example prompts"""
    ctx_triggered = ctx.triggered_id
    if not ctx_triggered:
        raise PreventUpdate
    
    examples = {
        'ex-1': 'A majestic dragon flying over mountains at sunset, fantasy art, detailed scales, epic',
        'ex-2': 'A futuristic cyberpunk city with neon lights and flying cars, rainy night, cinematic',
        'ex-3': 'An ancient medieval castle on a cliff overlooking the ocean, dramatic lighting, fantasy',
        'ex-4': 'An astronaut floating in space with Earth in the background, stars, nebula, photorealistic'
    }
    
    return examples.get(ctx_triggered, '')


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>AI Image Generator | Fast & Easy</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            }
            
            .card {
                border: none;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            
            .card-header {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: bold;
                border-radius: 12px 12px 0 0 !important;
            }
            
            .btn-primary {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                border: none;
                border-radius: 8px;
                font-weight: bold;
                transition: transform 0.2s;
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 15px rgba(102, 126, 234, 0.4);
            }
            
            .btn-success {
                background: linear-gradient(90deg, #00b09b 0%, #96c93d 100%);
                border: none;
                border-radius: 8px;
                font-weight: bold;
            }
            
            textarea {
                border-radius: 8px;
                border: 2px solid #e0e0e0;
                transition: border 0.3s;
            }
            
            textarea:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
                outline: none;
            }
            
            .progress-bar {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                border-radius: 4px;
            }
            
            .badge {
                font-weight: normal;
                font-size: 0.8em;
            }
            
            img {
                border: 8px solid white;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
        </style>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
            // Auto-resize textarea
            document.addEventListener('DOMContentLoaded', function() {
                const textarea = document.getElementById('prompt-input');
                if (textarea) {
                    // Set initial height
                    textarea.style.height = 'auto';
                    textarea.style.height = (textarea.scrollHeight) + 'px';
                    
                    // Auto-resize on input
                    textarea.addEventListener('input', function() {
                        this.style.height = 'auto';
                        this.style.height = (this.scrollHeight) + 'px';
                    });
                }
            });
        </script>
    </body>
</html>
'''


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 AI IMAGE GENERATOR - FIXED CUDA VERSION")
    print("="*60)
    print(f"📊 CUDA Available: {CUDA_AVAILABLE}")
    print(f"🎯 Device: {device}")
    print("="*60)
    print("✨ Features:")
    print("   • Fixed CUDA generator error")
    print("   • Real-time progress indicator")
    print("   • Auto negative prompts (no user input needed)")
    print("   • Fast generation (3-10 seconds)")
    print("="*60)
    print("📱 Open: http://localhost:8050")
    print("="*60)
    print("\n⚠️  Note: First generation may take 20-30 seconds to load model")
    print("   Subsequent generations will be fast!")
    print("="*60 + "\n")
    
    # Create output directory
    Path("generated_images").mkdir(exist_ok=True)
    
    app.run(
        host='0.0.0.0',
        port=8050,
        debug=True,
        dev_tools_hot_reload=True
    )