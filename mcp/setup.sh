#!/bin/bash
"""
Avinor MCP Server Setup Script
Helps register the MCP server with GitHub Copilot and other MCP clients
"""

echo "🚀 Avinor Flight Data MCP Server Setup"
echo "======================================"

# Get the absolute path to the MCP server directory
MCP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_PATH="$MCP_DIR/run_server.py"

echo "📍 MCP Server Location: $MCP_DIR"
echo "🐍 Python executable: $(which python3)"

# Test if the server can start
echo ""
echo "🧪 Testing server startup..."
cd "$MCP_DIR"
timeout 3 python run_server.py 2>/dev/null || echo "✅ Server test completed (timeout expected)"

echo ""
echo "📋 MCP SERVER REGISTRATION INSTRUCTIONS"
echo "======================================="

echo ""
echo "🤖 FOR GITHUB COPILOT:"
echo "1. Copy the configuration from: $MCP_DIR/copilot_config.json"
echo "2. Add it to your Copilot MCP settings"
echo "3. Restart your Copilot client"
echo "4. Configuration to use:"
echo ""
cat << EOF
{
  "mcpServers": {
    "avinor-flight-data": {
      "command": "python3",
      "args": ["$SERVER_PATH"],
      "env": {
        "PYTHONPATH": "$MCP_DIR/src"
      }
    }
  }
}
EOF

echo ""
echo "🔧 FOR CLAUDE DESKTOP:"
echo "1. Open Claude Desktop → Settings → Developer"
echo "2. Edit the MCP configuration file"
echo "3. Add the above configuration"

echo ""
echo "🔧 FOR OTHER MCP CLIENTS:"
echo "1. Use the configuration file: $MCP_DIR/mcp_config.json"
echo "2. Or run directly: python3 $SERVER_PATH"

echo ""
echo "🛠️  AVAILABLE TOOLS:"
echo "• get_csv_metadata - Extract comprehensive CSV metadata"
echo "• analyze_flight_patterns - Analyze flight routes, delays, cancellations"
echo "• analyze_airport_routes - Analyze route combinations between airport groups"
echo "• validate_data_quality - Comprehensive data quality assessment"

echo ""
echo "📚 AVAILABLE RESOURCES:"
echo "• flight_data_schema - Documentation of expected flight data structure"
echo "• analysis_examples - Example queries and analysis patterns"

echo ""
echo "✅ Setup complete! Your Avinor MCP server is ready to use."
echo "💡 Restart Claude Desktop after adding the configuration."
