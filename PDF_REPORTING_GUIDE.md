# PDF Report Generation - Implementation Guide

## Feature: C1 - PDF Report Generation & Export

**Priority:** Medium  
**Estimated Time:** 1-2 weeks  
**Dependencies:** Basic detection complete, reporting module exists

---

## 1. Requirements

### Functional Requirements
- [ ] Generate professional PDF reports from analysis results
- [ ] Include executive summary, detailed findings, and metadata
- [ ] Embed visualizations (spectrograms, waveforms, heatmaps)
- [ ] Add chain-of-custody information
- [ ] Support batch report generation (multiple analyses)
- [ ] Allow report customization (logo, branding, fields)
- [ ] Digital signature support (optional)

### Non-Functional Requirements
- [ ] Generate PDF in <5 seconds for standard report
- [ ] Support reports up to 50 pages
- [ ] PDF/A compliance for archival
- [ ] File size <10MB per report
- [ ] Printable at 300 DPI quality

### Report Sections Required
1. **Cover Page** - Title, timestamp, organization logo
2. **Executive Summary** - High-level verdict and confidence
3. **File Metadata** - Filename, size, duration, format
4. **Analysis Details** - Model version, processing time, settings
5. **Detection Results** - Verdict, confidence score, explanations
6. **Visualizations** - Spectrograms, waveforms, attention maps
7. **Technical Details** - Feature statistics, model outputs
8. **Chain of Custody** - Upload time, analyst, modifications
9. **Recommendations** - Next steps, security actions
10. **Appendix** - Glossary, methodology, references

---

## 2. Architecture

### Component Overview

```
┌─────────────────┐
│  Analysis Data  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Report Builder  │ (Aggregate data, format)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Template Engine │ (Jinja2 + HTML/CSS)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PDF Renderer    │ (WeasyPrint / ReportLab)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PDF Output    │
└─────────────────┘
```

### Technology Options

**Option 1: HTML → PDF (Recommended)**
- **Template:** Jinja2 (HTML + CSS)
- **Renderer:** WeasyPrint or wkhtmltopdf
- **Pros:** Easy styling, responsive, good for complex layouts
- **Cons:** Large dependencies

**Option 2: Python PDF Libraries**
- **Library:** ReportLab or FPDF2
- **Pros:** Lightweight, precise control, no HTML knowledge needed
- **Cons:** More code, harder to style

**Option 3: LaTeX → PDF**
- **Template:** LaTeX with Python bindings
- **Pros:** Publication quality, excellent typography
- **Cons:** Steep learning curve, complex setup

**Recommended:** Option 1 (HTML → PDF with WeasyPrint)

### Data Flow

1. **Fetch**: Retrieve analysis results from database
2. **Aggregate**: Combine data (metadata, verdict, visualizations)
3. **Render**: Apply Jinja2 template with data
4. **Convert**: WeasyPrint HTML → PDF
5. **Store**: Save PDF to storage (filesystem or S3)
6. **Serve**: Return PDF via download endpoint

---

## 3. Implementation Phases

### Phase 1: Report Data Aggregation (Days 1-2)

**Tasks:**
- [ ] Create report builder class
- [ ] Aggregate analysis data from database
- [ ] Generate summary statistics
- [ ] Prepare visualization data

**Code:**

```python
# services/api/app/pdf_report.py (NEW)

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import json

@dataclass
class ReportData:
    """Complete data package for PDF report."""
    
    # Header
    report_id: str
    generated_at: datetime
    organization: str
    analyst: Optional[str]
    
    # File info
    filename: str
    file_size: int
    media_type: str
    duration_sec: Optional[float]
    
    # Analysis
    analysis_id: str
    verdict: str
    confidence: float
    model_version: str
    processing_time_sec: float
    
    # Detailed results
    explanations: List[str]
    technical_indicators: dict
    risk_score: float
    
    # Visualizations (base64 encoded)
    spectrogram_image: Optional[str]
    waveform_image: Optional[str]
    heatmap_image: Optional[str]
    
    # Chain of custody
    upload_timestamp: datetime
    upload_ip: str
    upload_user: Optional[str]
    modifications: List[dict]
    
    # Recommendations
    recommended_actions: List[str]
    severity_level: str

class ReportBuilder:
    """Builds complete report data from analysis."""
    
    def __init__(self, db_session):
        self.db = db_session
    
    async def build_report(self, analysis_id: str) -> ReportData:
        """Aggregate all data needed for report."""
        
        # Fetch analysis
        analysis = await self.db.get_analysis(analysis_id)
        if not analysis:
            raise ValueError(f"Analysis {analysis_id} not found")
        
        # Fetch uploaded file metadata
        upload = await self.db.get_upload(analysis.upload_id)
        
        # Generate visualizations
        visualizations = await self.generate_visualizations(analysis)
        
        # Build recommendations
        recommendations = self.build_recommendations(
            analysis.verdict,
            analysis.confidence
        )
        
        return ReportData(
            report_id=str(uuid.uuid4()),
            generated_at=datetime.utcnow(),
            organization="Aegis-AI",
            analyst=None,  # TODO: Get from auth context
            
            filename=upload.original_filename,
            file_size=upload.size_bytes,
            media_type=upload.media_type,
            duration_sec=upload.duration_sec,
            
            analysis_id=str(analysis.id),
            verdict=analysis.verdict,
            confidence=analysis.confidence,
            model_version=analysis.model_version,
            processing_time_sec=analysis.processing_time_sec,
            
            explanations=analysis.explanations or [],
            technical_indicators=analysis.technical_details or {},
            risk_score=self.calculate_risk_score(analysis),
            
            spectrogram_image=visualizations.get("spectrogram"),
            waveform_image=visualizations.get("waveform"),
            heatmap_image=None,  # TODO: Implement
            
            upload_timestamp=upload.uploaded_at,
            upload_ip=upload.upload_ip or "N/A",
            upload_user=None,  # TODO: Get from auth
            modifications=[],
            
            recommended_actions=recommendations,
            severity_level=self.get_severity_level(analysis.verdict)
        )
    
    async def generate_visualizations(self, analysis) -> dict:
        """Generate visualization images for report."""
        import matplotlib.pyplot as plt
        import io
        import base64
        
        visualizations = {}
        
        # Load audio file
        audio_path = analysis.file_path
        if not Path(audio_path).exists():
            return visualizations
        
        # Generate spectrogram
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        
        # Spectrogram
        fig, ax = plt.subplots(figsize=(10, 4))
        spec = torchaudio.transforms.Spectrogram()(waveform)
        ax.imshow(spec.log().numpy()[0], aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title('Audio Spectrogram')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        visualizations["spectrogram"] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Waveform
        fig, ax = plt.subplots(figsize=(10, 3))
        time_axis = np.arange(waveform.shape[1]) / sr
        ax.plot(time_axis, waveform[0].numpy(), linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform')
        ax.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        visualizations["waveform"] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return visualizations
    
    def build_recommendations(self, verdict: str, confidence: float) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if verdict == "DEEPFAKE":
            recommendations.append("⚠️ HIGH PRIORITY: Do not trust this media")
            recommendations.append("Verify source through alternative channels")
            recommendations.append("Report to content platform if applicable")
            recommendations.append("Consider forensic analysis by specialists")
        elif verdict == "SUSPICIOUS":
            recommendations.append("⚠️ MEDIUM PRIORITY: Treat with caution")
            recommendations.append("Request original uncompressed file for re-analysis")
            recommendations.append("Cross-reference with other sources")
            recommendations.append("Monitor for additional manipulated content")
        else:  # AUTHENTIC
            if confidence < 0.8:
                recommendations.append("ℹ️ Confidence is moderate, consider secondary validation")
            recommendations.append("✓ Media appears authentic")
            recommendations.append("Verify context and distribution channel")
        
        recommendations.append("Archive this report for future reference")
        
        return recommendations
    
    def calculate_risk_score(self, analysis) -> float:
        """Calculate overall risk score (0-100)."""
        if analysis.verdict == "DEEPFAKE":
            base_score = 90
        elif analysis.verdict == "SUSPICIOUS":
            base_score = 60
        else:
            base_score = 20
        
        # Adjust based on confidence
        confidence_factor = (1 - analysis.confidence) * 20
        
        return min(100, base_score + confidence_factor)
    
    def get_severity_level(self, verdict: str) -> str:
        """Map verdict to severity level."""
        mapping = {
            "DEEPFAKE": "HIGH",
            "SUSPICIOUS": "MEDIUM",
            "AUTHENTIC": "LOW"
        }
        return mapping.get(verdict, "UNKNOWN")
```

**Testing:**
```python
# tests/test_report_builder.py

async def test_build_report_data(db_session, sample_analysis):
    builder = ReportBuilder(db_session)
    report_data = await builder.build_report(sample_analysis.id)
    
    assert report_data.report_id is not None
    assert report_data.verdict == sample_analysis.verdict
    assert report_data.filename is not None
    assert len(report_data.explanations) > 0
    assert len(report_data.recommended_actions) > 0

async def test_generate_visualizations(sample_audio_file):
    builder = ReportBuilder(None)
    viz = await builder.generate_visualizations(sample_audio_file)
    
    assert "spectrogram" in viz
    assert "waveform" in viz
    # Verify base64 encoded PNG
    assert viz["spectrogram"].startswith("iVBORw0KGgo")
```

### Phase 2: HTML Template Creation (Days 3-4)

**Tasks:**
- [ ] Design professional report template (HTML/CSS)
- [ ] Create Jinja2 templates for each section
- [ ] Add responsive styling
- [ ] Include print-specific CSS

**Template Structure:**

```
services/api/app/templates/
├── report_base.html          # Base template
├── sections/
│   ├── cover_page.html
│   ├── executive_summary.html
│   ├── file_metadata.html
│   ├── analysis_details.html
│   ├── detection_results.html
│   ├── visualizations.html
│   ├── technical_details.html
│   ├── chain_of_custody.html
│   ├── recommendations.html
│   └── appendix.html
└── styles/
    ├── report.css            # Main styles
    └── print.css             # Print-specific
```

**Base Template:**

```html
<!-- services/api/app/templates/report_base.html -->

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Aegis-AI Analysis Report - {{ report_id }}</title>
    <style>
        /* CSS for PDF rendering */
        @page {
            size: A4;
            margin: 2cm 1.5cm;
            
            @top-center {
                content: "Aegis-AI Deepfake Analysis Report";
                font-size: 10pt;
                color: #666;
            }
            
            @bottom-right {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 9pt;
                color: #999;
            }
        }
        
        body {
            font-family: 'Helvetica', 'Arial', sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
        }
        
        h1 {
            font-size: 24pt;
            color: #1a1a1a;
            border-bottom: 3px solid #007bff;
            padding-bottom: 0.5em;
            margin-top: 0;
        }
        
        h2 {
            font-size: 18pt;
            color: #2c3e50;
            margin-top: 2em;
            page-break-after: avoid;
        }
        
        h3 {
            font-size: 14pt;
            color: #34495e;
            margin-top: 1.5em;
        }
        
        .cover-page {
            page-break-after: always;
            text-align: center;
            padding-top: 30%;
        }
        
        .section {
            page-break-inside: avoid;
            margin-bottom: 2em;
        }
        
        .verdict-box {
            padding: 1.5em;
            border-radius: 8px;
            margin: 2em 0;
            font-size: 14pt;
            font-weight: bold;
        }
        
        .verdict-authentic {
            background-color: #d4edda;
            border: 2px solid #28a745;
            color: #155724;
        }
        
        .verdict-suspicious {
            background-color: #fff3cd;
            border: 2px solid #ffc107;
            color: #856404;
        }
        
        .verdict-deepfake {
            background-color: #f8d7da;
            border: 2px solid #dc3545;
            color: #721c24;
        }
        
        .metadata-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
        }
        
        .metadata-table th,
        .metadata-table td {
            padding: 0.75em;
            border: 1px solid #dee2e6;
            text-align: left;
        }
        
        .metadata-table th {
            background-color: #f8f9fa;
            font-weight: bold;
            width: 30%;
        }
        
        .visualization {
            max-width: 100%;
            page-break-inside: avoid;
            margin: 1em 0;
        }
        
        .visualization img {
            width: 100%;
            height: auto;
        }
        
        .recommendation {
            padding: 0.75em;
            margin: 0.5em 0;
            border-left: 4px solid #007bff;
            background-color: #f8f9fa;
        }
        
        .footer {
            margin-top: 3em;
            padding-top: 1em;
            border-top: 1px solid #dee2e6;
            font-size: 9pt;
            color: #666;
            text-align: center;
        }
    </style>
</head>
<body>
    {% include 'sections/cover_page.html' %}
    {% include 'sections/executive_summary.html' %}
    {% include 'sections/file_metadata.html' %}
    {% include 'sections/detection_results.html' %}
    {% include 'sections/visualizations.html' %}
    {% include 'sections/technical_details.html' %}
    {% include 'sections/chain_of_custody.html' %}
    {% include 'sections/recommendations.html' %}
    {% include 'sections/appendix.html' %}
    
    <div class="footer">
        <p>Generated by Aegis-AI Deepfake Detection System</p>
        <p>Report ID: {{ report_id }} | Generated: {{ generated_at }}</p>
        <p>This report is confidential and intended solely for authorized use.</p>
    </div>
</body>
</html>
```

**Executive Summary Section:**

```html
<!-- services/api/app/templates/sections/executive_summary.html -->

<div class="section executive-summary">
    <h1>Executive Summary</h1>
    
    <div class="verdict-box verdict-{{ verdict|lower }}">
        <div style="font-size: 20pt; margin-bottom: 0.5em;">
            {% if verdict == "AUTHENTIC" %}
                ✓ AUTHENTIC
            {% elif verdict == "SUSPICIOUS" %}
                ⚠ SUSPICIOUS
            {% else %}
                ✗ DEEPFAKE DETECTED
            {% endif %}
        </div>
        <div style="font-size: 14pt; font-weight: normal;">
            Confidence: {{ (confidence * 100)|round(1) }}%
        </div>
        <div style="font-size: 14pt; font-weight: normal;">
            Risk Score: {{ risk_score|round(0) }}/100
        </div>
    </div>
    
    <h3>Summary</h3>
    <p>
        This report presents the results of automated deepfake detection analysis 
        performed on <strong>{{ filename }}</strong>. The analysis was conducted on 
        {{ generated_at.strftime('%B %d, %Y at %H:%M UTC') }} using 
        Aegis-AI model version {{ model_version }}.
    </p>
    
    <p>
        <strong>Verdict:</strong> The media has been classified as <strong>{{ verdict }}</strong> 
        with a confidence level of <strong>{{ (confidence * 100)|round(1) }}%</strong>.
    </p>
    
    {% if explanations %}
    <h3>Key Findings</h3>
    <ul>
        {% for explanation in explanations %}
        <li>{{ explanation }}</li>
        {% endfor %}
    </ul>
    {% endif %}
</div>
```

### Phase 3: PDF Generation Engine (Days 5-6)

**Tasks:**
- [ ] Integrate WeasyPrint
- [ ] Implement PDF generation service
- [ ] Add error handling
- [ ] Test with various report sizes

**Code:**

```python
# services/api/app/pdf_generator.py (NEW)

from weasyprint import HTML, CSS
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PDFGenerator:
    """Generate PDF reports from templates."""
    
    def __init__(self, template_dir: str = "app/templates"):
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
    
    async def generate_report(self, report_data: ReportData, output_path: Path) -> Path:
        """Generate PDF report from ReportData."""
        
        try:
            # Render HTML from template
            template = self.env.get_template('report_base.html')
            html_content = template.render(
                report_id=report_data.report_id,
                generated_at=report_data.generated_at,
                organization=report_data.organization,
                analyst=report_data.analyst,
                
                filename=report_data.filename,
                file_size=self.format_file_size(report_data.file_size),
                media_type=report_data.media_type,
                duration_sec=report_data.duration_sec,
                
                analysis_id=report_data.analysis_id,
                verdict=report_data.verdict,
                confidence=report_data.confidence,
                model_version=report_data.model_version,
                processing_time_sec=report_data.processing_time_sec,
                
                explanations=report_data.explanations,
                technical_indicators=report_data.technical_indicators,
                risk_score=report_data.risk_score,
                
                spectrogram_image=report_data.spectrogram_image,
                waveform_image=report_data.waveform_image,
                heatmap_image=report_data.heatmap_image,
                
                upload_timestamp=report_data.upload_timestamp,
                upload_ip=report_data.upload_ip,
                upload_user=report_data.upload_user,
                modifications=report_data.modifications,
                
                recommended_actions=report_data.recommended_actions,
                severity_level=report_data.severity_level
            )
            
            # Convert HTML to PDF
            logger.info(f"Generating PDF report: {output_path}")
            
            HTML(string=html_content).write_pdf(
                output_path,
                stylesheets=[
                    CSS(string='''
                        @page { 
                            size: A4; 
                            margin: 2cm 1.5cm;
                        }
                    ''')
                ]
            )
            
            logger.info(f"PDF report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            raise
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
```

**Installation:**
```bash
# services/api/requirements.txt
weasyprint==60.0
Jinja2==3.1.2
```

**Testing:**
```python
# tests/test_pdf_generator.py

async def test_generate_pdf_report(sample_report_data, tmp_path):
    generator = PDFGenerator()
    output_path = tmp_path / "test_report.pdf"
    
    result = await generator.generate_report(sample_report_data, output_path)
    
    assert result.exists()
    assert result.stat().st_size > 0
    
    # Verify it's a valid PDF
    with open(result, 'rb') as f:
        assert f.read(4) == b'%PDF'

def test_pdf_contains_correct_data(generated_pdf_path, sample_report_data):
    # Use PyPDF2 to extract text and verify content
    import PyPDF2
    
    with open(generated_pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    
    assert sample_report_data.filename in text
    assert sample_report_data.verdict in text
    assert str(sample_report_data.confidence) in text
```

### Phase 4: API Endpoint & Integration (Days 7-8)

**Tasks:**
- [ ] Add PDF download endpoint
- [ ] Implement caching (don't regenerate same report)
- [ ] Add batch report generation
- [ ] Update API documentation

**Code:**

```python
# services/api/app/main.py (UPDATE)

from app.pdf_report import ReportBuilder
from app.pdf_generator import PDFGenerator

@app.get("/v1/analysis/{analysis_id}/report/pdf", response_class=FileResponse)
async def download_pdf_report(
    analysis_id: UUID,
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Generate and download PDF report for an analysis."""
    
    # Check if report already exists (caching)
    cache_path = Path(f"reports/{analysis_id}.pdf")
    if cache_path.exists():
        return FileResponse(
            cache_path,
            media_type="application/pdf",
            filename=f"aegis_report_{analysis_id}.pdf"
        )
    
    # Build report data
    builder = ReportBuilder(db)
    try:
        report_data = await builder.build_report(str(analysis_id))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    # Generate PDF
    generator = PDFGenerator()
    cache_path.parent.mkdir(exist_ok=True)
    
    try:
        pdf_path = await generator.generate_report(report_data, cache_path)
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail="PDF generation failed")
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"aegis_report_{analysis_id}.pdf",
        headers={"Content-Disposition": f"attachment; filename=aegis_report_{analysis_id}.pdf"}
    )

@app.post("/v1/reports/batch", response_class=FileResponse)
async def generate_batch_report(
    analysis_ids: List[UUID],
    db: Session = Depends(get_db),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Generate a combined PDF report for multiple analyses."""
    
    # TODO: Implement batch reporting
    # Aggregate multiple analyses into single report
    raise HTTPException(status_code=501, detail="Batch reporting not yet implemented")
```

**Testing:**
```python
# tests/test_pdf_endpoint.py

def test_download_pdf_report(client, sample_analysis_id):
    response = client.get(f"/v1/analysis/{sample_analysis_id}/report/pdf")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "aegis_report" in response.headers["content-disposition"]
    
    # Verify PDF content
    assert response.content[:4] == b'%PDF'

def test_pdf_report_caching(client, sample_analysis_id):
    # First request generates PDF
    response1 = client.get(f"/v1/analysis/{sample_analysis_id}/report/pdf")
    time1 = response1.elapsed.total_seconds()
    
    # Second request should use cache (faster)
    response2 = client.get(f"/v1/analysis/{sample_analysis_id}/report/pdf")
    time2 = response2.elapsed.total_seconds()
    
    assert time2 < time1 * 0.5  # Cache should be 2x faster
    assert response1.content == response2.content
```

### Phase 5: Testing & Polish (Days 9-10)

**Tasks:**
- [ ] Visual QA on reports
- [ ] Test with various analysis types (audio, video)
- [ ] Verify print quality
- [ ] Performance optimization
- [ ] Documentation

---

## 4. Testing Strategy

### Unit Tests (30+ tests)
- Report data aggregation
- Template rendering
- PDF generation
- File size formatting
- Risk score calculation

### Integration Tests (15+ tests)
- End-to-end PDF generation
- API endpoint responses
- Caching behavior
- Error handling

### Visual Tests (Manual)
- Print quality check
- Layout consistency
- Image rendering
- Typography
- Color accuracy

### Performance Tests (5+ tests)
- Generation time (<5s)
- Memory usage
- File size (<10MB)
- Concurrent generation

---

## 5. Acceptance Criteria

### Definition of Done
- [ ] PDF generation working for all analysis types
- [ ] Professional, printable reports
- [ ] Generation time <5 seconds
- [ ] All visualizations rendering correctly
- [ ] Caching implemented
- [ ] All tests passing (50+ tests)
- [ ] API documentation updated

### Quality Gates
- [ ] Reports pass visual QA
- [ ] Print at 300 DPI quality
- [ ] PDF/A compliant
- [ ] File size <10MB

---

## 6. Future Enhancements

1. **Digital Signatures:** Add cryptographic signatures for legal validity
2. **Customization:** Allow users to customize branding, logo, colors
3. **Internationalization:** Support multiple languages
4. **Interactive PDFs:** Add form fields, annotations
5. **Report Templates:** Multiple template options (brief, detailed, technical)

---

## 7. Resources

**Libraries:**
- WeasyPrint: https://weasyprint.org/
- Jinja2: https://jinja.palletsprojects.com/
- ReportLab (alternative): https://www.reportlab.com/

**PDF Standards:**
- PDF/A (archival): https://en.wikipedia.org/wiki/PDF/A
- Digital signatures: ISO 32000-2

---

**Estimated Effort:** 80-100 hours (1 engineer, 2 weeks)  
**Team:** 1 Backend Engineer with design skills
