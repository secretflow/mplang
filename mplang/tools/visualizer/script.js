document.addEventListener('DOMContentLoaded', function() {
    console.log('Visualizer script loaded.');

    let cy; // Cytoscape instance
    cytoscape.use(cytoscapeElk);
    cytoscape.use(cytoscapeSvg);

    // --- DOM Element References ---
    const fileInput = document.getElementById('ir-file-input');
    const container = document.getElementById('cy-container');
    const exportPngBtn = document.getElementById('export-png');
    const exportSvgBtn = document.getElementById('export-svg');
    const exportJsonBtn = document.getElementById('export-json');

    // --- Event Listeners ---
    fileInput.addEventListener('change', handleFileSelect);
    exportPngBtn.addEventListener('click', () => exportGraph('png'));
    exportSvgBtn.addEventListener('click', () => exportGraph('svg'));
    exportJsonBtn.addEventListener('click', () => exportGraph('json'));

    const partyColors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4'];
    const securityOps = {
        'quote_gen': { color: '#f032e6', shape: 'diamond' },
        'quote_verify': { color: '#f032e6', shape: 'star' },
        'reveal': { color: '#a9a9a9', shape: 'vee' }
    };

    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                parseAndRenderGraph(JSON.parse(e.target.result));
            } catch (error) {
                console.error('Error processing file:', error);
                alert('Failed to process file. See console.');
            }
        };
        reader.readAsText(file);
    }

    function exportGraph(format) {
        if (!cy) {
            alert('Please load a graph first.');
            return;
        }
        let content, fileName, mimeType;
        switch (format) {
            case 'png':
                content = cy.png({ output: 'blob', full: true });
                fileName = 'graph.png';
                download(content, fileName);
                break;
            case 'svg':
                content = cy.svg({ full: true });
                fileName = 'graph.svg';
                download(new Blob([content], { type: 'image/svg+xml' }), fileName);
                break;
            case 'json':
                content = JSON.stringify(cy.json(), null, 2);
                fileName = 'graph.json';
                download(new Blob([content], { type: 'application/json' }), fileName);
                break;
        }
    }

    function download(blob, fileName) {
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = fileName;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(link.href);
    }

    function makeTooltip(node) {
        const { id, op_type, party, attrs, risk } = node.data();
        let content = `<strong>ID:</strong> ${id}<br><strong>Op:</strong> ${op_type}<br><strong>Party:</strong> ${party}`;
        if (risk) content += `<br><strong style="color: red;">Risk:</strong> ${risk}`;
        if (attrs && Object.keys(attrs).length > 0) {
            content += '<hr><strong>Attributes:</strong><br>';
            for (const [key, value] of Object.entries(attrs)) {
                let attrValue = JSON.stringify(value);
                if (attrValue.length > 50) attrValue = attrValue.substring(0, 50) + '...';
                content += `&bull; ${key}: ${attrValue}<br>`;
            }
        }
        return tippy(node.popperRef(), { content, trigger: 'manual', allowHTML: true, theme: 'light-border' }).instances[0];
    }

    function analyzeAndHighlightRisks(cy) {
        cy.nodes('[op_type = "reveal"]').addClass('risk-reveal');
        const teeParties = new Set(cy.nodes('[op_type = "quote_gen"]').map(n => n.data('party')));
        if (teeParties.size === 0) return;
        cy.edges().forEach(edge => {
            const sourceNode = edge.source(), targetNode = edge.target();
            const sourceParty = sourceNode.data('party'), targetParty = targetNode.data('party');
            if (sourceParty !== targetParty && teeParties.has(targetParty)) {
                const verifiers = cy.nodes(`[party = "${sourceParty}"][op_type = "quote_verify"]`);
                if (verifiers.length === 0) {
                    edge.addClass('risk-unverified-transfer');
                    sourceNode.data('risk', 'Sends data to TEE without any verification');
                }
            }
        });
    }

    function parseAndRenderGraph(irGraph) {
        if (!irGraph || !irGraph.nodes) return;
        const elements = [], nodeIds = new Set(irGraph.nodes.map(n => n.name)), parties = new Set();
        irGraph.nodes.forEach(node => {
            let party = 'unknown';
            if (node.outs_info && node.outs_info[0] && node.outs_info[0].pmask != null) {
                party = `P${node.outs_info[0].pmask}`;
                parties.add(party);
            }
            elements.push({ group: 'nodes', data: { id: node.name, op_type: node.op_type, party: party, parent: party, attrs: node.attrs || {} } });
        });
        Array.from(parties).forEach(partyId => {
            elements.push({ group: 'nodes', data: { id: partyId }, classes: 'compound-parent' });
        });
        irGraph.nodes.forEach(node => {
            if (node.inputs) {
                node.inputs.forEach((input, i) => {
                    const source = input.split(':')[0];
                    if (nodeIds.has(source)) {
                        elements.push({ group: 'edges', data: { id: `${source}_to_${node.name}_${i}`, source, target: node.name } });
                    }
                });
            }
        });
        if (cy) cy.destroy();
        cy = cytoscape({
            container, elements,
            style: [
                { selector: 'node', style: { 'label': 'data(op_type)', 'font-size': '10px', 'text-valign': 'center', 'text-halign': 'center', 'color': 'white', 'text-outline-color': '#555', 'text-outline-width': 1 } },
                { selector: '.compound-parent', style: { 'background-color': '#f0f0f0', 'border-color': '#ccc', 'border-width': 2, 'label': 'data(id)', 'font-size': '14px', 'font-weight': 'bold', 'text-valign': 'top', 'color': '#000', 'text-outline-width': 0 } },
                { selector: 'edge', style: { 'width': 2, 'line-color': '#ccc', 'target-arrow-color': '#ccc', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier', 'transition-property': 'line-color', 'transition-duration': '0.2s' } },
                ...Array.from(parties).map((partyId, i) => ({ selector: `node[parent="${partyId}"]`, style: { 'background-color': partyColors[i % partyColors.length] } })),
                ...Object.entries(securityOps).map(([op, style]) => ({ selector: `node[op_type="${op}"]`, style: { 'background-color': style.color, 'shape': style.shape } })),
                { selector: '.risk-reveal', style: { 'border-color': 'gold', 'border-width': 3 } },
                { selector: '.risk-unverified-transfer', style: { 'line-color': 'red', 'target-arrow-color': 'red', 'line-style': 'dashed' } }
            ],
            layout: { name: 'elk', elk: { algorithm: 'layered', 'elk.direction': 'DOWN' }, fit: true, padding: 50 }
        });
        analyzeAndHighlightRisks(cy);
        cy.nodes().filter(n => !n.isParent()).forEach(node => {
            const tooltip = makeTooltip(node);
            node.on('mouseover', () => tooltip.show());
            node.on('mouseout', () => tooltip.hide());
        });
    }
});
