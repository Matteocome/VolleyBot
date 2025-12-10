/*
 * Backend Server for Volleyball Email Checker
 * 
 * Setup:
 * npm install express cors imap mailparser dotenv
 * 
 * Create .env file with:
 * EMAIL_USER=your_email@gmail.com
 * EMAIL_PASSWORD=your_app_password
 * PORT=3001
 */

const express = require('express');
const cors = require('cors');
const Imap = require('imap');
const { simpleParser } = require('mailparser');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public')); // Serve static files

// Configuration
const CONFIG = {
    email: {
        user: process.env.EMAIL_USER,
        password: process.env.EMAIL_PASSWORD,
        host: 'imap.gmail.com',
        port: 993,
        tls: true
    },
    senderEmail: 'contact@gomammoth.co.uk'
};

// ==================== API ENDPOINT ====================
app.post('/api/check-volleyball-email', async (req, res) => {
    console.log('📧 API called: Checking for volleyball emails...');
    
    try {
        const result = await checkForVolleyballEmails();
        
        if (result) {
            res.json({
                success: true,
                matchDetails: result,
                message: 'Match details found!'
            });
        } else {
            res.json({
                success: false,
                matchDetails: null,
                message: 'No new volleyball emails found'
            });
        }
    } catch (error) {
        console.error('❌ Error:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', message: 'Server is running' });
});

// ==================== EMAIL CHECKING LOGIC ====================
function checkForVolleyballEmails() {
    return new Promise((resolve, reject) => {
        const imap = new Imap({
            user: CONFIG.email.user,
            password: CONFIG.email.password,
            host: CONFIG.email.host,
            port: CONFIG.email.port,
            tls: CONFIG.email.tls,
            tlsOptions: { rejectUnauthorized: false }
        });

        imap.once('ready', () => {
            console.log('✅ Connected to Gmail');
            
            imap.openBox('INBOX', false, (err, box) => {
                if (err) {
                    reject(err);
                    return;
                }

                // Search for unread emails from GO Mammoth from the last 7 days
                const sevenDaysAgo = new Date();
                sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);

                imap.search(['UNSEEN', ['FROM', CONFIG.senderEmail], ['SINCE', sevenDaysAgo]], (err, results) => {
                    if (err) {
                        reject(err);
                        return;
                    }

                    if (results.length === 0) {
                        console.log('📭 No new volleyball emails');
                        imap.end();
                        resolve(null);
                        return;
                    }

                    console.log(`📧 Found ${results.length} email(s) from GO Mammoth`);
                    
                    // Get the most recent email
                    const latestEmail = [results[results.length - 1]];
                    const fetch = imap.fetch(latestEmail, { bodies: '' });

                    fetch.on('message', (msg) => {
                        msg.on('body', (stream) => {
                            simpleParser(stream, (err, parsed) => {
                                if (err) {
                                    reject(err);
                                    return;
                                }

                                const matchDetails = parseVolleyballEmail(parsed.text);
                                
                                if (matchDetails) {
                                    console.log('✅ Match details extracted:', matchDetails);
                                    resolve(matchDetails);
                                } else {
                                    console.log('⚠️ Could not parse match details');
                                    console.log('Email text:', parsed.text.substring(0, 500));
                                    resolve(null);
                                }
                            });
                        });
                    });

                    fetch.once('end', () => {
                        imap.end();
                    });
                });
            });
        });

        imap.once('error', (err) => {
            console.error('❌ IMAP error:', err);
            reject(err);
        });

        imap.connect();
    });
}

// ==================== PARSE EMAIL ====================
function parseVolleyballEmail(emailText) {
    console.log('🔍 Parsing email...');
    
    // Try multiple patterns
    const patterns = [
        // Pattern 1: Standard format
        /Your next fixture is (.+?) vs (.+?), at (\d{1,2}:\d{2}) on (.+?) at (.+?)\./i,
        
        // Pattern 2: Without comma before "at"
        /Your next fixture is (.+?) vs (.+?) at (\d{1,2}:\d{2}) on (.+?) at (.+?)\./i,
        
        // Pattern 3: More flexible
        /fixture is (.+?) vs (.+?)[,\s]+at (\d{1,2}:\d{2})[,\s]+on (.+?) at (.+?)[\.\s]/i,
        
        // Pattern 4: Very flexible
        /(.+?)\s+vs\s+(.+?)[,\s]+at\s+(\d{1,2}:\d{2})[,\s]+on\s+(.+?)\s+at\s+(.+?)[\.\s]/i
    ];
    
    for (let i = 0; i < patterns.length; i++) {
        const match = emailText.match(patterns[i]);
        
        if (match) {
            console.log(`✅ Matched with pattern ${i + 1}`);
            
            const details = {
                team1: match[1].trim(),
                team2: match[2].trim(),
                time: match[3].trim(),
                date: match[4].trim(),
                venue: match[5].trim()
            };
            
            // Validate
            if (details.team1 && details.team2 && details.time && details.venue) {
                return details;
            }
        }
    }
    
    console.log('❌ No pattern matched');
    return null;
}

// ==================== START SERVER ====================
app.listen(PORT, () => {
    console.log('='.repeat(60));
    console.log('🏐 Volleyball Email Checker Server');
    console.log('='.repeat(60));
    console.log(`✅ Server running on http://localhost:${PORT}`);
    console.log(`📧 Email: ${CONFIG.email.user}`);
    console.log(`🔍 Monitoring: ${CONFIG.senderEmail}`);
    console.log('='.repeat(60));
    console.log('\n📱 Open http://localhost:${PORT} in your browser');
    console.log('🔘 Click "Check Email" button to check for new matches\n');
});